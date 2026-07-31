#!/usr/bin/env bash
# benchmark.sh — measure GA wall-time vs number of workers.
set -euo pipefail

cd "$(dirname "$0")"

# ---- Config ------------------------------------------------------------
WORKER_COUNTS=(1 2 4 8 12)   # worker counts to sweep
RUNS_PER_COUNT=1             # bump to 3+ for averaging (handles noise)
WARMUP=0                     # set to 1 to discard the first run per count
RESULTS_CSV="benchmark_results.csv"
# ------------------------------------------------------------------------

echo "workers,run,wall_s,orchestrator_s,best_fitness" > "$RESULTS_CSV"

# Build once so image build time isn't counted in the sweep.
echo ">> Building images (once)..."
docker compose build

for n in "${WORKER_COUNTS[@]}"; do
  for run in $(seq 1 "$RUNS_PER_COUNT"); do
    echo ""
    echo "=============================="
    echo "  N_WORKERS=$n   run=$run"
    echo "=============================="

    # Clean slate so Envoy DNS / container state doesn't leak between runs.
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true

    export N_WORKERS=$n
    LOG=$(mktemp)
    start=$(date +%s.%N)

    # --abort-on-container-exit: stop everything once orchestrator finishes.
    # --exit-code-from orchestrator: surface orchestrator's exit status.
    set +e
    docker compose up \
      --abort-on-container-exit \
      --exit-code-from orchestrator \
      --scale worker="$n" \
      > "$LOG" 2>&1
    rc=$?
    set -e

    end=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.2f\", $end - $start}")

    if [ $rc -ne 0 ]; then
      echo "  !! run failed (exit $rc). Tail of log:"
      tail -n 20 "$LOG"
      echo "$n,$run,$wall,," >> "$RESULTS_CSV"
      rm -f "$LOG"
      continue
    fi

    # Parse the orchestrator's own reported elapsed + best fitness.
    orch_time=$(grep -oE 'Finished GA run in [0-9.]+' "$LOG" \
                | grep -oE '[0-9.]+' | tail -1 || echo "NA")
    best=$(grep -oE 'Best fitness: [-0-9.]+' "$LOG" \
           | grep -oE '[-0-9.]+' | tail -1 || echo "NA")

    echo "  wall=${wall}s   orchestrator=${orch_time}s   best=${best}"
    echo "$n,$run,$wall,$orch_time,$best" >> "$RESULTS_CSV"

    # Optional warmup discard
    if [ "$WARMUP" -eq 1 ] && [ "$run" -eq 1 ]; then
      echo "  (warmup run discarded)"
      # remove last line we just wrote
      sed -i '$d' "$RESULTS_CSV"
    fi

    rm -f "$LOG"
  done
done

echo ""
echo "=============================="
echo "  Results"
echo "=============================="
column -t -s, "$RESULTS_CSV"
