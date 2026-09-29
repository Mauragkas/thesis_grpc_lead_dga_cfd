#!/usr/bin/env bash
#
# Runs a true 2-node distributed ring migration benchmark with identical
# parameters as the single-island runs (POP_SIZE=100, MAX_GENERATIONS=50),
# across 2 distinct seed pairs.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORC_DIR="${REPO_ROOT}/orchestrator"
DATA_DIR="${EVAL_DIR}/data"
FIG_DIR="${EVAL_DIR}/figures"
ORC_BIN="${ORC_DIR}/target/release/orchestrator"

mkdir -p "${DATA_DIR}" "${FIG_DIR}"

POP_SIZE=100
MAX_GENS=50
MIG_INTERVAL=5
MIG_COUNT=2
EVAL_EP="127.0.0.1:50051"

# Check if worker is up
if ! nc -z 127.0.0.1 50051 2>/dev/null; then
  echo "Error: Worker is not running on 127.0.0.1:50051!"
  echo "Start the worker first: cd ../../worker && uv run python src/worker/main.py"
  exit 1
fi

if [ ! -f "${ORC_BIN}" ]; then
  echo "Building orchestrator in release mode..."
  cargo build --release --manifest-path "${ORC_DIR}/Cargo.toml"
fi

echo "=========================================================="
echo " Starting 2-Node Ring Migration Benchmark"
echo " Condition: POP_SIZE=${POP_SIZE}, MAX_GENS=${MAX_GENS}, MIG_INTERVAL=${MIG_INTERVAL}"
echo "=========================================================="

# Pair 1: Seeds (42, 101)
echo ">> [Pair 1/2] Starting Node 1 (Seed 42) & Node 2 (Seed 101)..."
EVAL_ENDPOINT="${EVAL_EP}" \
RING_BIND="0.0.0.0:50060" \
RING_SELF_ADDRESS="127.0.0.1:50060" \
GA_EXPORT_PATH="${DATA_DIR}/multi_island_node1_seed42.jsonl" \
GA_RECORD_POPULATION=true \
GA_SEED=42 \
MAX_GENERATIONS="${MAX_GENS}" \
POP_SIZE="${POP_SIZE}" \
MIGRATION_INTERVAL="${MIG_INTERVAL}" \
MIGRATION_COUNT="${MIG_COUNT}" \
"${ORC_BIN}" &
PID1=$!

sleep 2

EVAL_ENDPOINT="${EVAL_EP}" \
RING_BIND="0.0.0.0:50061" \
RING_SELF_ADDRESS="127.0.0.1:50061" \
RING_BOOTSTRAP="http://127.0.0.1:50060" \
GA_EXPORT_PATH="${DATA_DIR}/multi_island_node2_seed101.jsonl" \
GA_RECORD_POPULATION=true \
GA_SEED=101 \
MAX_GENERATIONS="${MAX_GENS}" \
POP_SIZE="${POP_SIZE}" \
MIGRATION_INTERVAL="${MIG_INTERVAL}" \
MIGRATION_COUNT="${MIG_COUNT}" \
"${ORC_BIN}" &
PID2=$!

wait $PID1
wait $PID2
echo ">> Pair 1 completed!"

# Pair 2: Seeds (777, 999)
echo ">> [Pair 2/2] Starting Node 1 (Seed 777) & Node 2 (Seed 999)..."
EVAL_ENDPOINT="${EVAL_EP}" \
RING_BIND="0.0.0.0:50060" \
RING_SELF_ADDRESS="127.0.0.1:50060" \
GA_EXPORT_PATH="${DATA_DIR}/multi_island_node1_seed777.jsonl" \
GA_RECORD_POPULATION=true \
GA_SEED=777 \
MAX_GENERATIONS="${MAX_GENS}" \
POP_SIZE="${POP_SIZE}" \
MIGRATION_INTERVAL="${MIG_INTERVAL}" \
MIGRATION_COUNT="${MIG_COUNT}" \
"${ORC_BIN}" &
PID3=$!

sleep 2

EVAL_ENDPOINT="${EVAL_EP}" \
RING_BIND="0.0.0.0:50061" \
RING_SELF_ADDRESS="127.0.0.1:50061" \
RING_BOOTSTRAP="http://127.0.0.1:50060" \
GA_EXPORT_PATH="${DATA_DIR}/multi_island_node2_seed999.jsonl" \
GA_RECORD_POPULATION=true \
GA_SEED=999 \
MAX_GENERATIONS="${MAX_GENS}" \
POP_SIZE="${POP_SIZE}" \
MIGRATION_INTERVAL="${MIG_INTERVAL}" \
MIGRATION_COUNT="${MIG_COUNT}" \
"${ORC_BIN}" &
PID4=$!

wait $PID3
wait $PID4
echo ">> Pair 2 completed!"

echo "=========================================================="
echo " Re-generating Figure 1.2 and Figure 1.3..."
echo "=========================================================="
python3 "${SCRIPT_DIR}/plot_single_vs_multi_island.py" \
  --single-pattern "${DATA_DIR}/single_island_seed*.jsonl" \
  --multi-pattern "${DATA_DIR}/multi_island_node*.jsonl" \
  --out-dir "${FIG_DIR}"

python3 "${SCRIPT_DIR}/plot_diversity_spread.py" \
  --single-pattern "${DATA_DIR}/single_island_seed*.jsonl" \
  --multi-pattern "${DATA_DIR}/multi_island_node*.jsonl" \
  --out-dir "${FIG_DIR}"

echo "Done! Updated figures saved to ${FIG_DIR}"
