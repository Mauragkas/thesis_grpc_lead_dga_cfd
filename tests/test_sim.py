#!/usr/bin/env python3
"""
Similarity query evaluation for LEAD DHT with multi-probe Hilbert keys.

1. Clear all stale keys from previous runs
2. Insert N plane configs, indexed under NUM_CURVES rotated Hilbert curves
3. Verify all probe keys are present and unique in the cluster
4. Pick a target config
5. Compute ground-truth Euclidean nearest neighbors
6. Verify the DHT /range mechanism returns Hilbert-ordered slices per curve
7. Similarity queries:
   a. Single-curve forward range
   b. Single-curve wider range + client re-rank
   c. Multi-probe wider range + merge + re-rank (all curves)
   d. Multi-probe bidirectional + merge + re-rank
   e. Random baseline
8. Summary: recall, avg distance, quality vs random
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).parent / "../worker/src/worker"))
sys.path.insert(0, str(Path(__file__).parent / "../worker/src"))
sys.path.insert(0, str(Path(__file__).parent / "../worker"))
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
            r = requests.post(f"{url}/kv/{encoded}", data=value, timeout=5)
            if r.status_code in (200, 201):
                return True
        except requests.RequestException:
            continue
    return False


def get_routed(key: str) -> tuple[bool, str | None]:
    encoded = quote(key, safe="")
    for url in DHT_NODES:
        try:
            r = requests.get(f"{url}/kv/{encoded}", timeout=5)
            if r.status_code == 200:
                return True, r.text
        except requests.RequestException:
            continue
    return False, None


def range_query(start_key: str, count: int) -> list[tuple[str, str]]:
    encoded = quote(start_key, safe="")
    for url in DHT_NODES:
        try:
            r = requests.get(f"{url}/range?key={encoded}&count={count}", timeout=30)
            if r.status_code == 200:
                return [(k, v) for k, v in r.json()]
        except requests.RequestException:
            continue
    return []


def clear_all_keys() -> int:
    deleted = 0
    for url in DHT_NODES:
        try:
            r = requests.get(f"{url}/keys", timeout=5)
            if r.status_code != 200:
                continue
            for key, _ in r.json():
                encoded = quote(key, safe="")
                try:
                    requests.delete(f"{url}/kv/local/{encoded}", timeout=5)
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
            r = requests.get(f"{url}/keys", timeout=5)
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
    # ── Step 1: Clear stale keys ─────────────────────────────────────
    print("=== Clearing stale keys from previous runs ===")
    cleared = clear_all_keys()
    print(f"Cleared {cleared} keys\n")
    if cleared > 0:
        print("Waiting 5s for DHT to settle after clearing...")
        time.sleep(5)

    remaining = get_cluster_keys()
    print("  Cluster is clean.\n" if not remaining else f"  WARNING: {len(remaining)} keys still present\n")

    # ── Step 2: Generate and insert configs ──────────────────────────
    total_probes = N_PLANES * NUM_CURVES
    print(f"=== Generating {N_PLANES} configs x {NUM_CURVES} Hilbert curves ({total_probes} keys) ===\n")
    configs = [make_plane(i) for i in range(N_PLANES)]

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

    print("Waiting 30s for DHT to stabilize and balance...")
    time.sleep(30)

    # ── Step 3: Verify cluster contents ──────────────────────────────
    print("\nKey distribution:")
    total_seen = 0
    for port in [2001, 2002, 2003]:
        try:
            resp = requests.get(f"http://localhost:{port}/keys", timeout=5)
            resp.raise_for_status()
            total_seen += len(resp.json())
            print(f"  node {port}: {len(resp.json())} keys")
        except requests.RequestException:
            print(f"  node {port}: unreachable")

    cluster_keys = get_cluster_keys()
    print(f"  total appearances: {total_seen}")
    print(f"  unique keys:       {len(cluster_keys)}")

    if cluster_keys != expected_keys:
        missing = sorted(expected_keys - cluster_keys)
        extra = sorted(cluster_keys - expected_keys)
        if missing:
            print(f"\n  Missing keys: {len(missing)} (first 3: {[k[:24] for k in missing[:3]]})")
        if extra:
            print(f"  Extra keys:   {len(extra)} (first 3: {[k[:24] for k in extra[:3]]})")
        fail("cluster key set does not match inserted keys")

    print("  All keys present and no stale keys. ✓\n")

    # ── Step 4: Target ───────────────────────────────────────────────
    target_idx = N_PLANES // 2
    target_cfg = configs[target_idx]

    print(f"\n{'='*64}")
    print(f"  Target config (index {target_idx})")
    print(f"{'='*64}")
    print(f"  wing_span:       {target_cfg['wing_span']:.1f}")
    print(f"  fuse_length:     {target_cfg['fuse_length']:.1f}")
    print(f"  wing_sweep:      {target_cfg['wing_sweep']:.1f}")
    print(f"  wing_root_chord: {target_cfg['wing_root_chord']:.1f}")

    # ── Step 5: Ground truth ─────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Ground Truth: Top {K_NEIGHBORS} Euclidean Nearest Neighbors")
    print(f"{'='*64}")

    distances = []
    for i, cfg in enumerate(configs):
        if i == target_idx:
            continue
        distances.append((euclidean_distance(target_cfg, cfg), i, cfg))
    distances.sort(key=lambda x: x[0])

    ground_truth = distances[:K_NEIGHBORS]
    print(f"{'Rank':<6} {'Idx':<6} {'Dist':<10} {'wing_span':<12} {'fuse_len':<12} {'sweep':<8}")
    print("-" * 60)
    for rank, (d, idx, cfg) in enumerate(ground_truth):
        print(f"{rank+1:<6} {idx:<6} {d:<10.4f} {cfg['wing_span']:<12.1f} {cfg['fuse_length']:<12.1f} {cfg['wing_sweep']:<8.1f}")

    gt_avg = sum(d for d, _, _ in ground_truth) / len(ground_truth)
    gt_indices = {idx for _, idx, _ in ground_truth}
    print(f"\n  Avg distance: {gt_avg:.4f}")

    # ── Step 6: Per-curve mechanism check ────────────────────────────
    print(f"\n{'='*64}")
    print(f"  DHT /range mechanism check (per curve, count={K_NEIGHBORS + 1})")
    print(f"{'='*64}")

    curve_order: list[list[str]] = []
    curve_pos: list[dict[str, int]] = []
    all_mechanism_ok = True
    for c in range(NUM_CURVES):
        local_keys = sorted(config_key(cfg, c) for cfg in configs)
        curve_order.append(local_keys)
        curve_pos.append({k: i for i, k in enumerate(local_keys)})

        tkey = config_key(target_cfg, c)
        pos = curve_pos[c][tkey]
        expected_slice = local_keys[pos + 1 : pos + K_NEIGHBORS + 1]
        raw = range_query(tkey, K_NEIGHBORS + 1)
        got = [k for k, _ in raw if k != tkey][:K_NEIGHBORS]

        ok = got == expected_slice
        all_mechanism_ok = all_mechanism_ok and ok
        print(f"  curve {c}: {'✓ /range returns exact Hilbert slice' if ok else '✗ mismatch'}")
        if ok and expected_slice:
            first_idx = int(json.loads(range_query(expected_slice[0], 1)[0][1])["plane_idx"])
            print(f"           first result: idx {first_idx}, dist {euclidean_distance(target_cfg, configs[first_idx]):.4f}")

    if not all_mechanism_ok:
        fail("DHT /range did not return the expected Hilbert-ordered slice on every curve")

    # ── Step 7a: Single-curve forward ────────────────────────────────
    tkey0 = config_key(target_cfg, 0)
    raw = range_query(tkey0, K_NEIGHBORS + 1)
    single_forward: list[tuple[float, int, dict[str, Any]]] = []
    for k, v in raw:
        if k == tkey0:
            continue
        try:
            cfg = parse_hilbert_key(k)
            idx = int(json.loads(v).get("plane_idx", -1))
            single_forward.append((euclidean_distance(target_cfg, cfg), idx, cfg))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    single_forward = single_forward[:K_NEIGHBORS]

    print(f"\n{'='*64}")
    print(f"  Single Curve 0 Forward (count={K_NEIGHBORS + 1})")
    print(f"{'='*64}")
    print_results_table("", single_forward, gt_indices)
    sf_avg = sum(d for d, _, _ in single_forward) / len(single_forward) if single_forward else float("inf")
    print(f"\n  Avg distance: {sf_avg:.4f}  |  Overlap: {len(gt_indices & {i for _, i, _ in single_forward})}/{K_NEIGHBORS}")

    # ── Step 7b: Single-curve wider + re-rank ────────────────────────
    wide_count = K_NEIGHBORS * WIDER_FACTOR + 1
    raw = range_query(tkey0, wide_count)
    candidates: dict[int, tuple[float, dict[str, Any]]] = {}
    for k, v in raw:
        if k != tkey0:
            add_candidate(candidates, target_cfg, k, v)
    single_wider = sorted(candidates.values(), key=lambda x: x[0])[:K_NEIGHBORS]
    single_wider = [(d, idx, cfg) for idx, (d, cfg) in enumerate(candidates.items())]
    # fix ordering to sorted list
    single_wider = [(d, idx, cfg) for idx, (d, cfg) in sorted(candidates.items(), key=lambda kv: kv[1][0])[:K_NEIGHBORS]]

    print(f"\n{'='*64}")
    print(f"  Single Curve 0 Wider + Re-rank (count={wide_count})")
    print(f"{'='*64}")
    print_results_table("", single_wider, gt_indices)
    sw_avg = sum(d for d, _, _ in single_wider) / len(single_wider) if single_wider else float("inf")
    print(f"\n  Avg distance: {sw_avg:.4f}  |  Overlap: {len(gt_indices & {i for _, i, _ in single_wider})}/{K_NEIGHBORS}")

    # ── Step 7c: Multi-probe forward + re-rank ───────────────────────
    mp_candidates: dict[int, tuple[float, dict[str, Any]]] = {}
    for c in range(NUM_CURVES):
        tkey = config_key(target_cfg, c)
        for k, v in range_query(tkey, wide_count):
            if k != tkey:
                add_candidate(mp_candidates, target_cfg, k, v)
    mp_forward = [(d, idx, cfg) for idx, (d, cfg) in sorted(mp_candidates.items(), key=lambda kv: kv[1][0])[:K_NEIGHBORS]]

    print(f"\n{'='*64}")
    print(f"  Multi-Probe ({NUM_CURVES} curves) Forward + Re-rank")
    print(f"{'='*64}")
    print_results_table("", mp_forward, gt_indices)
    mpf_avg = sum(d for d, _, _ in mp_forward) / len(mp_forward) if mp_forward else float("inf")
    print(f"\n  Avg distance: {mpf_avg:.4f}  |  Overlap: {len(gt_indices & {i for _, i, _ in mp_forward})}/{K_NEIGHBORS}")

    # ── Step 7d: Multi-probe bidirectional + re-rank ────────────────
    for c in range(NUM_CURVES):
        tkey = config_key(target_cfg, c)
        pos = curve_pos[c][tkey]
        for k, v in range_query(tkey, wide_count):
            if k != tkey:
                add_candidate(mp_candidates, target_cfg, k, v)
        if pos > 0:
            prev_key = curve_order[c][pos - 1]
            for k, v in range_query(prev_key, wide_count):
                if curve_pos[c].get(k, -1) < pos:
                    add_candidate(mp_candidates, target_cfg, k, v)

    mp_bi = [(d, idx, cfg) for idx, (d, cfg) in sorted(mp_candidates.items(), key=lambda kv: kv[1][0])[:K_NEIGHBORS]]

    print(f"\n{'='*64}")
    print(f"  Multi-Probe ({NUM_CURVES} curves) Bidirectional + Re-rank")
    print(f"{'='*64}")
    print_results_table("", mp_bi, gt_indices)
    mpb_avg = sum(d for d, _, _ in mp_bi) / len(mp_bi) if mp_bi else float("inf")
    print(f"\n  Avg distance: {mpb_avg:.4f}  |  Overlap: {len(gt_indices & {i for _, i, _ in mp_bi})}/{K_NEIGHBORS}")

    # ── Step 8: Exact lookups ────────────────────────────────────────
    print(f"\n{'='*64}")
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

    # ── Step 9: Random baseline ──────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Random Baseline ({K_NEIGHBORS} random configs)")
    print(f"{'='*64}")

    rng = random.Random(42)
    random_pool = [i for i in range(N_PLANES) if i != target_idx]
    rng.shuffle(random_pool)
    random_results = []
    for i in random_pool[:K_NEIGHBORS]:
        d = euclidean_distance(target_cfg, configs[i])
        random_results.append((d, i, configs[i]))

    random_avg = sum(d for d, _, _ in random_results) / len(random_results)
    print(f"  Avg distance: {random_avg:.4f}")

    # ── Step 10: Summary ─────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*64}")
    print(f"  Ground truth avg:                     {gt_avg:.4f}")
    print(f"  Single forward avg:                   {sf_avg:.4f}")
    print(f"  Single wider+rerank avg:              {sw_avg:.4f}")
    print(f"  Multi-probe forward avg:              {mpf_avg:.4f}")
    print(f"  Multi-probe bidirectional avg:        {mpb_avg:.4f}")
    print(f"  Random avg:                           {random_avg:.4f}")
    print(f"  {'─'*56}")

    rows = [
        ("Single forward", single_forward, sf_avg),
        ("Single wider+rerank", single_wider, sw_avg),
        ("Multi-probe forward", mp_forward, mpf_avg),
        ("Multi-probe bidirectional", mp_bi, mpb_avg),
    ]
    for name, res, avg in rows:
        recall = len(gt_indices & {i for _, i, _ in res}) / K_NEIGHBORS
        q = quality_score(avg, gt_avg, random_avg)
        ratio = avg / gt_avg if gt_avg > 0 else float("inf")
        print(f"  {name:<26} recall={recall:.0%}  quality={q:.0%}  ratio={ratio:.2f}x")

    print(f"{'='*64}")
    print("\nNote: Keys are indexed under NUM_CURVES rotated Hilbert curves.")
    print("      /range returns exact Hilbert-ordered slices per curve.")
    print("      Multi-probe merges candidates across curves, deduplicates by")
    print("      plane_idx, and re-ranks by Euclidean distance. This recovers")
    print("      neighbors that sit behind the target or far away in any single")
    print("      curve's 1D order (the curse of dimensionality in 10+D).")
    print("      Storage scales as NUM_CURVES x N keys; query cost as")
    print("      NUM_CURVES x 2 x (WIDER_FACTOR x K) routed range scans.")


if __name__ == "__main__":
    main()
