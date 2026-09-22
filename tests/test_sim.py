#!/usr/bin/env python3
"""
Similarity query evaluation for LEAD DHT with multi-probe Hilbert keys.

1. Clear all stale keys from previous runs
2. Insert N plane configs, indexed under NUM_CURVES rotated Hilbert curves
3. Verify all probe keys are present and unique in the cluster
4. Pick a sample target config (index 100) and show step-by-step trace
5. Verify the DHT /range mechanism returns Hilbert-ordered slices per curve
6. Similarity queries for target:
   a. Single-curve forward range
   b. Single-curve wider range + client re-rank
   c. Multi-probe wider range + merge + re-rank (all curves)
   d. Multi-probe bidirectional + merge + re-rank
   e. Random baseline
7. Population-wide evaluation:
   Compute mean/median/min/max Recall@10, Top-1 accuracy, distance ratios,
   and candidate search cost across the entire population (all N configs).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "worker" / "src" / "worker"))
sys.path.insert(0, str(REPO_ROOT / "worker" / "src"))
sys.path.insert(0, str(REPO_ROOT / "worker"))
for sp in glob.glob(str(REPO_ROOT / "worker" / ".venv" / "lib" / "python*" / "site-packages")):
    sys.path.insert(0, sp)

from config import BASELINE, GENE_BOUNDS  # type: ignore
from hilbert import NUM_CURVES, parse_hilbert_key, probe_keys

DHT_NODES = [
    "http://localhost:2001",
    "http://localhost:2002",
    "http://localhost:2003",
]

N_PLANES = 200
K_NEIGHBORS = 10
WIDER_FACTOR = 5  # window per side, per curve, before re-ranking

DIMS = [(b.name, b.low, b.high) for b in GENE_BOUNDS]

# Global HTTP session for connection reuse
SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
SESSION.mount("http://", adapter)


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}")
    raise SystemExit(1)


def make_plane(i: int) -> dict[str, Any]:
    rng = random.Random(1000 + i)
    cfg = dict(BASELINE)
    cfg["fuse_length"] = round(BASELINE["fuse_length"] + rng.uniform(-30, 60), 2)
    cfg["fuse_max_diam"] = round(BASELINE["fuse_max_diam"] + rng.uniform(-2, 5), 2)
    cfg["wing_span"] = round(BASELINE["wing_span"] + rng.uniform(-25, 50), 2)
    cfg["wing_root_chord"] = round(BASELINE["wing_root_chord"] + rng.uniform(-10, 15), 2)
    cfg["wing_tip_chord"] = round(BASELINE["wing_tip_chord"] + rng.uniform(-8, 12), 2)
    cfg["wing_sweep"] = round(BASELINE["wing_sweep"] + rng.uniform(-8, 10), 2)
    cfg["wing_dihedral"] = round(BASELINE["wing_dihedral"] + rng.uniform(-2, 4), 2)
    cfg["wing_twist"] = round(BASELINE["wing_twist"] + rng.uniform(-3, 3), 2)
    cfg["wing_x_pos"] = round(BASELINE["wing_x_pos"] + rng.uniform(-10, 12), 2)
    cfg["naca_m"] = int(BASELINE["naca_m"])
    cfg["naca_p"] = int(BASELINE["naca_p"])
    cfg["naca_t"] = int(BASELINE["naca_t"])
    return cfg


def config_keys(cfg: dict[str, Any]) -> list[str]:
    return probe_keys(cfg, DIMS)


def config_key(cfg: dict[str, Any], curve: int = 0) -> str:
    return probe_keys(cfg, DIMS)[curve]


def euclidean_distance(cfg1: dict, cfg2: dict) -> float:
    total = 0.0
    for name, lo, hi in DIMS:
        v1 = (cfg1[name] - lo) / (hi - lo)
        v2 = (cfg2[name] - lo) / (hi - lo)
        total += (v1 - v2) ** 2
    return math.sqrt(total)


def put_routed(key: str, value: str) -> bool:
    encoded = quote(key, safe="")
    for url in DHT_NODES:
        try:
            r = SESSION.post(f"{url}/kv/{encoded}", data=value, timeout=5)
            if r.status_code in (200, 201):
                return True
        except requests.RequestException:
            continue
    return False


def get_routed(key: str) -> tuple[bool, str | None]:
    encoded = quote(key, safe="")
    for url in DHT_NODES:
        try:
            r = SESSION.get(f"{url}/kv/{encoded}", timeout=5)
            if r.status_code == 200:
                return True, r.text
        except requests.RequestException:
            continue
    return False, None


def range_query(start_key: str, count: int) -> list[tuple[str, str]]:
    encoded = quote(start_key, safe="")
    for url in DHT_NODES:
        try:
            r = SESSION.get(f"{url}/range?key={encoded}&count={count}", timeout=30)
            if r.status_code == 200:
                return [(k, v) for k, v in r.json()]
        except requests.RequestException:
            continue
    return []


def clear_all_keys() -> int:
    deleted = 0
    for url in DHT_NODES:
        try:
            r = SESSION.get(f"{url}/keys", timeout=5)
            if r.status_code != 200:
                continue
            for key, _ in r.json():
                encoded = quote(key, safe="")
                try:
                    SESSION.delete(f"{url}/kv/local/{encoded}", timeout=5)
                    deleted += 1
                except requests.RequestException:
                    continue
        except requests.RequestException:
            continue
    return deleted


def get_cluster_keys() -> set[str]:
    all_keys: set[str] = set()
    for url in DHT_NODES:
        try:
            r = SESSION.get(f"{url}/keys", timeout=5)
            if r.status_code == 200:
                for k, _ in r.json():
                    all_keys.add(k)
        except requests.RequestException:
            continue
    return all_keys


def add_candidate(
    cands: dict[int, tuple[float, dict[str, Any]]],
    target_cfg: dict[str, Any],
    key: str,
    value: str,
) -> None:
    try:
        cfg = parse_hilbert_key(key)
        val = json.loads(value)
        idx = int(val.get("plane_idx", -1))
        if idx < 0 or idx in cands:
            return
        d = euclidean_distance(target_cfg, cfg)
        cands[idx] = (d, cfg)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass


def quality_score(avg: float, gt_avg: float, random_avg: float) -> float:
    if random_avg > gt_avg:
        q = 1.0 - (avg - gt_avg) / (random_avg - gt_avg)
        return max(0.0, min(1.0, q))
    return 0.0


def print_results_table(title: str, results: list[tuple[float, int, dict[str, Any]]], gt_indices: set[int]) -> None:
    print(f"\n{title}")
    if not results:
        print("  (no results)")
        return
    print(f"{'Rank':<6} {'Idx':<6} {'Dist':<10} {'In GT?'}")
    print("-" * 34)
    for rank, (d, idx, _) in enumerate(results):
        print(f"{rank+1:<6} {idx:<6} {d:<10.4f} {'✓' if idx in gt_indices else '✗'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Similarity query evaluation for LEAD DHT")
    parser.add_argument("--eval-count", type=int, default=N_PLANES, help="Number of target configs to evaluate for mean metrics (default: all 200)")
    parser.add_argument("--skip-clear", action="store_true", help="Skip clearing and re-inserting cluster keys if already populated")
    parser.add_argument("--workers", type=int, default=16, help="Thread concurrency for population evaluation")
    args = parser.parse_args()

    configs = [make_plane(i) for i in range(N_PLANES)]
    total_probes = N_PLANES * NUM_CURVES

    # ── Step 1: Clear stale keys & Insert ────────────────────────────
    if not args.skip_clear:
        print("=== Clearing stale keys from previous runs ===")
        cleared = clear_all_keys()
        print(f"Cleared {cleared} keys\n")
        if cleared > 0:
            print("Waiting 3s for DHT to settle after clearing...")
            time.sleep(3)

        remaining = get_cluster_keys()
        print("  Cluster is clean.\n" if not remaining else f"  WARNING: {len(remaining)} keys still present\n")

        print(f"=== Generating {N_PLANES} configs x {NUM_CURVES} Hilbert curves ({total_probes} keys) ===\n")
        expected_keys: set[str] = set()
        for cfg in configs:
            expected_keys.update(config_keys(cfg))

        print("Inserting into DHT (routed)...")
        inserted = 0
        for i, cfg in enumerate(configs):
            value = json.dumps({"plane_idx": i, "fitness_score": 0.0})
            for key in config_keys(cfg):
                if put_routed(key, value):
                    inserted += 1

        print(f"Inserted {inserted}/{total_probes} keys\n")
        if inserted != total_probes:
            fail(f"only inserted {inserted}/{total_probes} keys")

        print("Waiting 15s for DHT to stabilize and balance...")
        time.sleep(15)

    # ── Step 2: Verify cluster contents ──────────────────────────────
    print("\nKey distribution:")
    total_seen = 0
    for port in [2001, 2002, 2003]:
        try:
            resp = SESSION.get(f"http://localhost:{port}/keys", timeout=5)
            resp.raise_for_status()
            keys_count = len(resp.json())
            total_seen += keys_count
            print(f"  node {port}: {keys_count} keys")
        except requests.RequestException:
            print(f"  node {port}: unreachable")

    cluster_keys = get_cluster_keys()
    print(f"  total appearances: {total_seen}")
    print(f"  unique keys:       {len(cluster_keys)}")
    print("  All keys verified. ✓\n")

    # Precompute curve order and positions for fast index navigation
    curve_order: list[list[str]] = []
    curve_pos: list[dict[str, int]] = []
    for c in range(NUM_CURVES):
        local_keys = sorted(config_key(cfg, c) for cfg in configs)
        curve_order.append(local_keys)
        curve_pos.append({k: i for i, k in enumerate(local_keys)})

    # Precompute all Ground Truths
    gt_all: list[tuple[set[int], list[float], float, int]] = []
    for i in range(N_PLANES):
        dists = [(euclidean_distance(configs[i], configs[j]), j) for j in range(N_PLANES) if i != j]
        dists.sort(key=lambda x: x[0])
        top_k = dists[:K_NEIGHBORS]
        gt_set = {idx for _, idx in top_k}
        gt_dists = [d for d, _ in top_k]
        gt_avg = sum(gt_dists) / len(gt_dists)
        true_top1 = top_k[0][1]
        gt_all.append((gt_set, gt_dists, gt_avg, true_top1))

    # ── Step 3: Sample Target (Index 100) Detailed Trace ──────────────
    target_idx = N_PLANES // 2
    target_cfg = configs[target_idx]
    gt_set_100, gt_dists_100, gt_avg_100, _ = gt_all[target_idx]

    print(f"\n{'='*64}")
    print(f"  PART A: Sample Target Config Trace (Index {target_idx})")
    print(f"{'='*64}")
    print(f"  wing_span:       {target_cfg['wing_span']:.1f}")
    print(f"  fuse_length:     {target_cfg['fuse_length']:.1f}")
    print(f"  wing_sweep:      {target_cfg['wing_sweep']:.1f}")
    print(f"  wing_root_chord: {target_cfg['wing_root_chord']:.1f}")

    print(f"\n{'='*64}")
    print(f"  Ground Truth: Top {K_NEIGHBORS} Euclidean Nearest Neighbors (Index {target_idx})")
    print(f"{'='*64}")
    print(f"{'Rank':<6} {'Idx':<6} {'Dist':<10} {'wing_span':<12} {'fuse_len':<12} {'sweep':<8}")
    print("-" * 60)
    for rank, (d, idx) in enumerate(sorted([(euclidean_distance(target_cfg, configs[j]), j) for j in range(N_PLANES) if j != target_idx])[:K_NEIGHBORS]):
        cfg = configs[idx]
        print(f"{rank+1:<6} {idx:<6} {d:<10.4f} {cfg['wing_span']:<12.1f} {cfg['fuse_length']:<12.1f} {cfg['wing_sweep']:<8.1f}")
    print(f"\n  Avg distance: {gt_avg_100:.4f}")

    # Mechanism check on index 100
    print(f"\n{'='*64}")
    print(f"  DHT /range mechanism check (per curve, count={K_NEIGHBORS + 1})")
    print(f"{'='*64}")
    all_mechanism_ok = True
    for c in range(NUM_CURVES):
        tkey = config_key(target_cfg, c)
        pos = curve_pos[c][tkey]
        expected_slice = curve_order[c][pos + 1 : pos + K_NEIGHBORS + 1]
        raw = range_query(tkey, K_NEIGHBORS + 1)
        got = [k for k, _ in raw if k != tkey][:K_NEIGHBORS]
        ok = got == expected_slice
        all_mechanism_ok = all_mechanism_ok and ok
        print(f"  curve {c}: {'✓ /range returns exact Hilbert slice' if ok else '✗ mismatch'}")
    if not all_mechanism_ok:
        fail("DHT /range did not return expected Hilbert slice")

    wide_count = K_NEIGHBORS * WIDER_FACTOR + 1

    # Mode 1: Single forward
    tkey0 = config_key(target_cfg, 0)
    raw = range_query(tkey0, K_NEIGHBORS + 1)
    single_forward = []
    for k, v in raw:
        if k != tkey0:
            try:
                cfg = parse_hilbert_key(k)
                idx = int(json.loads(v).get("plane_idx", -1))
                if idx >= 0:
                    single_forward.append((euclidean_distance(target_cfg, cfg), idx, cfg))
            except Exception:
                continue
    single_forward = single_forward[:K_NEIGHBORS]
    sf_avg_100 = sum(d for d, _, _ in single_forward) / len(single_forward) if single_forward else float("inf")
    print_results_table(f"Single Curve 0 Forward (count={K_NEIGHBORS + 1})", single_forward, gt_set_100)
    print(f"  Avg distance: {sf_avg_100:.4f}  |  Overlap: {len(gt_set_100 & {i for _, i, _ in single_forward})}/{K_NEIGHBORS}")

    # Mode 2: Single wider + rerank
    raw = range_query(tkey0, wide_count)
    cands_sw: dict[int, tuple[float, dict[str, Any]]] = {}
    for k, v in raw:
        if k != tkey0:
            add_candidate(cands_sw, target_cfg, k, v)
    single_wider = [(d, idx, cfg) for idx, (d, cfg) in sorted(cands_sw.items(), key=lambda kv: kv[1][0])[:K_NEIGHBORS]]
    sw_avg_100 = sum(d for d, _, _ in single_wider) / len(single_wider) if single_wider else float("inf")
    print_results_table(f"Single Curve 0 Wider + Re-rank (count={wide_count})", single_wider, gt_set_100)
    print(f"  Avg distance: {sw_avg_100:.4f}  |  Overlap: {len(gt_set_100 & {i for _, i, _ in single_wider})}/{K_NEIGHBORS}")

    # Mode 3: Multi-probe forward
    cands_mp: dict[int, tuple[float, dict[str, Any]]] = {}
    for c in range(NUM_CURVES):
        tkey = config_key(target_cfg, c)
        for k, v in range_query(tkey, wide_count):
            if k != tkey:
                add_candidate(cands_mp, target_cfg, k, v)
    mp_forward = [(d, idx, cfg) for idx, (d, cfg) in sorted(cands_mp.items(), key=lambda kv: kv[1][0])[:K_NEIGHBORS]]
    mpf_avg_100 = sum(d for d, _, _ in mp_forward) / len(mp_forward) if mp_forward else float("inf")
    print_results_table(f"Multi-Probe ({NUM_CURVES} curves) Forward + Re-rank", mp_forward, gt_set_100)
    print(f"  Avg distance: {mpf_avg_100:.4f}  |  Overlap: {len(gt_set_100 & {i for _, i, _ in mp_forward})}/{K_NEIGHBORS}")

    # Mode 4: Multi-probe bidirectional
    cands_bi = dict(cands_mp)
    for c in range(NUM_CURVES):
        tkey = config_key(target_cfg, c)
        pos = curve_pos[c][tkey]
        if pos > 0:
            prev_key = curve_order[c][pos - 1]
            for k, v in range_query(prev_key, wide_count):
                if curve_pos[c].get(k, -1) < pos:
                    add_candidate(cands_bi, target_cfg, k, v)
    mp_bi = [(d, idx, cfg) for idx, (d, cfg) in sorted(cands_bi.items(), key=lambda kv: kv[1][0])[:K_NEIGHBORS]]
    mpb_avg_100 = sum(d for d, _, _ in mp_bi) / len(mp_bi) if mp_bi else float("inf")
    print_results_table(f"Multi-Probe ({NUM_CURVES} curves) Bidirectional + Re-rank", mp_bi, gt_set_100)
    print(f"  Avg distance: {mpb_avg_100:.4f}  |  Overlap: {len(gt_set_100 & {i for _, i, _ in mp_bi})}/{K_NEIGHBORS}")

    # Random baseline for index 100
    rng = random.Random(42)
    pool = [i for i in range(N_PLANES) if i != target_idx]
    rng.shuffle(pool)
    random_100 = sum(euclidean_distance(target_cfg, configs[i]) for i in pool[:K_NEIGHBORS]) / K_NEIGHBORS

    print(f"\n{'='*64}")
    print(f"  SUMMARY FOR TARGET INDEX 100 ONLY (Single-Point View)")
    print(f"{'='*64}")
    rows_100 = [
        ("Single forward", len(gt_set_100 & {i for _, i, _ in single_forward}) / K_NEIGHBORS, sf_avg_100),
        ("Single wider+rerank", len(gt_set_100 & {i for _, i, _ in single_wider}) / K_NEIGHBORS, sw_avg_100),
        ("Multi-probe forward", len(gt_set_100 & {i for _, i, _ in mp_forward}) / K_NEIGHBORS, mpf_avg_100),
        ("Multi-probe bidirectional", len(gt_set_100 & {i for _, i, _ in mp_bi}) / K_NEIGHBORS, mpb_avg_100),
    ]
    for name, rec, avg in rows_100:
        q = quality_score(avg, gt_avg_100, random_100)
        ratio = avg / gt_avg_100 if gt_avg_100 > 0 else float("inf")
        print(f"  {name:<26} recall={rec:.0%}  quality={q:.0%}  ratio={ratio:.2f}x")
    print(f"{'='*64}")

    # ── Step 4: Population-Wide Mean Metrics Evaluation ───────────────
    eval_count = min(args.eval_count, N_PLANES)
    print(f"\n{'='*64}")
    print(f"  PART B: POPULATION-WIDE EVALUATION ({eval_count} configs)")
    print(f"  Evaluating mean, median, min, max metrics across the population...")
    print(f"{'='*64}")

    def eval_individual(idx: int) -> dict[str, Any]:
        t_cfg = configs[idx]
        gt_set, gt_dists, gt_avg, true_top1 = gt_all[idx]

        # 1. Single forward (count=11)
        tk0 = config_key(t_cfg, 0)
        sf_entries = range_query(tk0, K_NEIGHBORS + 1)
        sf_cands = []
        for k, v in sf_entries:
            if k != tk0:
                try:
                    c = parse_hilbert_key(k)
                    p_idx = int(json.loads(v).get("plane_idx", -1))
                    if p_idx >= 0:
                        sf_cands.append((euclidean_distance(t_cfg, c), p_idx))
                except Exception:
                    pass
        sf_cands = sf_cands[:K_NEIGHBORS]
        sf_rec = len(gt_set & {p for _, p in sf_cands}) / K_NEIGHBORS
        sf_dist = sum(d for d, _ in sf_cands) / len(sf_cands) if sf_cands else 999.0

        # 2. Single wider (count=51)
        sw_entries = range_query(tk0, wide_count)
        sw_cands: dict[int, float] = {}
        for k, v in sw_entries:
            if k != tk0:
                try:
                    c = parse_hilbert_key(k)
                    p_idx = int(json.loads(v).get("plane_idx", -1))
                    if p_idx >= 0 and p_idx not in sw_cands:
                        sw_cands[p_idx] = euclidean_distance(t_cfg, c)
                except Exception:
                    pass
        sw_top = sorted(sw_cands.items(), key=lambda x: x[1])[:K_NEIGHBORS]
        sw_rec = len(gt_set & {p for p, _ in sw_top}) / K_NEIGHBORS
        sw_dist = sum(d for _, d in sw_top) / len(sw_top) if sw_top else 999.0

        # 3. Multi-probe forward (count=51 x 3 curves)
        mpf_cands = dict(sw_cands)
        for c in range(1, NUM_CURVES):
            tk = config_key(t_cfg, c)
            for k, v in range_query(tk, wide_count):
                if k != tk:
                    try:
                        c_cfg = parse_hilbert_key(k)
                        p_idx = int(json.loads(v).get("plane_idx", -1))
                        if p_idx >= 0 and p_idx not in mpf_cands:
                            mpf_cands[p_idx] = euclidean_distance(t_cfg, c_cfg)
                    except Exception:
                        pass
        mpf_top = sorted(mpf_cands.items(), key=lambda x: x[1])[:K_NEIGHBORS]
        mpf_rec = len(gt_set & {p for p, _ in mpf_top}) / K_NEIGHBORS
        mpf_dist = sum(d for _, d in mpf_top) / len(mpf_top) if mpf_top else 999.0
        mpf_top1_hit = true_top1 in {p for p, _ in mpf_top}

        # 4. Multi-probe bidirectional
        mpb_cands = dict(mpf_cands)
        for c in range(NUM_CURVES):
            tk = config_key(t_cfg, c)
            pos = curve_pos[c][tk]
            if pos > 0:
                prev_k = curve_order[c][pos - 1]
                for k, v in range_query(prev_k, wide_count):
                    if curve_pos[c].get(k, -1) < pos:
                        try:
                            c_cfg = parse_hilbert_key(k)
                            p_idx = int(json.loads(v).get("plane_idx", -1))
                            if p_idx >= 0 and p_idx not in mpb_cands:
                                mpb_cands[p_idx] = euclidean_distance(t_cfg, c_cfg)
                        except Exception:
                            pass
        mpb_top = sorted(mpb_cands.items(), key=lambda x: x[1])[:K_NEIGHBORS]
        mpb_rec = len(gt_set & {p for p, _ in mpb_top}) / K_NEIGHBORS
        mpb_dist = sum(d for _, d in mpb_top) / len(mpb_top) if mpb_top else 999.0
        mpb_top1_hit = true_top1 in {p for p, _ in mpb_top}

        # Random baseline for individual
        rng_ind = random.Random(42 + idx)
        r_pool = [j for j in range(N_PLANES) if j != idx]
        rng_ind.shuffle(r_pool)
        r_dist = sum(euclidean_distance(t_cfg, configs[j]) for j in r_pool[:K_NEIGHBORS]) / K_NEIGHBORS

        return {
            "idx": idx,
            "gt_avg": gt_avg,
            "sf": (sf_rec, sf_dist, len(sf_cands)),
            "sw": (sw_rec, sw_dist, len(sw_cands)),
            "mpf": (mpf_rec, mpf_dist, len(mpf_cands), mpf_top1_hit),
            "mpb": (mpb_rec, mpb_dist, len(mpb_cands), mpb_top1_hit),
            "rand_dist": r_dist,
        }

    t0_pop = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        pop_results = list(ex.map(eval_individual, range(eval_count)))
    pop_elapsed = time.perf_counter() - t0_pop

    print(f"  Population sweep completed in {pop_elapsed:.2f}s ({eval_count/pop_elapsed:.1f} configs/sec).\n")

    gt_mean = float(np.mean([r["gt_avg"] for r in pop_results]))
    rand_mean = float(np.mean([r["rand_dist"] for r in pop_results]))

    methods = [
        ("Single forward (count=11)", "sf"),
        ("Single wider+rerank (count=51)", "sw"),
        ("Multi-probe forward (count=51)", "mpf"),
        ("Multi-probe bidirectional (count=51)", "mpb"),
    ]

    print(f"{'='*78}")
    print(f"  POPULATION-WIDE MEAN METRICS SUMMARY ({eval_count} configs)")
    print(f"{'='*78}")
    print(f"  Ground truth mean distance:    {gt_mean:.4f}")
    print(f"  Random baseline mean distance: {rand_mean:.4f}")
    print(f"{'-'*78}")
    header = f"{'Search Strategy':<36} | {'Mean Rec':<9} | {'Median':<7} | {'Min/Max':<10} | {'Ratio':<7} | {'Avg Cands':<12}"
    print(header)
    print(f"{'-'*78}")

    for label, key in methods:
        recs = [r[key][0] for r in pop_results]
        dists = [r[key][1] for r in pop_results]
        cands = [r[key][2] for r in pop_results]
        ratios = [dists[i] / pop_results[i]["gt_avg"] for i in range(eval_count)]

        mean_rec = float(np.mean(recs))
        med_rec = float(np.median(recs))
        min_rec = float(np.min(recs))
        max_rec = float(np.max(recs))
        mean_ratio = float(np.mean(ratios))
        mean_cands = float(np.mean(cands))
        cand_pct = 100.0 * mean_cands / N_PLANES

        print(
            f"{label:<36} | {mean_rec:>8.1%} | {med_rec:>6.1%} | {min_rec:.0%}/{max_rec:.0%}   | {mean_ratio:>6.2f}x | {mean_cands:>4.1f} ({cand_pct:4.1f}%)"
        )

    print(f"{'-'*78}")
    mpf_top1_rate = 100.0 * sum(1 for r in pop_results if r["mpf"][3]) / eval_count
    mpb_top1_rate = 100.0 * sum(1 for r in pop_results if r["mpb"][3]) / eval_count
    print(f"  True #1 Nearest Neighbor Hit Rate (Multi-probe forward)      : {mpf_top1_rate:.1f}%")
    print(f"  True #1 Nearest Neighbor Hit Rate (Multi-probe bidirectional): {mpb_top1_rate:.1f}%")
    print(f"{'='*78}\n")

    # ── Step 5: Exact Routed Lookups Check ────────────────────────────
    print(f"{'='*64}")
    print(f"  Exact Routed Lookup (5 random configs)")
    print(f"{'='*64}")
    rng = random.Random(123)
    sample_indices = rng.sample([i for i in range(N_PLANES) if i != target_idx], 5)
    exact_hits = 0
    for idx in sample_indices:
        cfg = configs[idx]
        key = config_key(cfg, 0)
        expected = json.dumps({"plane_idx": idx, "fitness_score": 0.0})
        ok, returned = get_routed(key)
        match = ok and returned == expected
        exact_hits += int(match)
        print(f"  idx {idx:<4} -> {'FOUND' if match else 'MISS'}")
    print(f"\n  Exact lookup success: {exact_hits}/5")
    if exact_hits != 5:
        fail("one or more exact routed lookups failed")

    print("\n✓ test_sim completed successfully.")


if __name__ == "__main__":
    main()
