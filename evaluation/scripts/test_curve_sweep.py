#!/usr/bin/env python3
"""
Multi-Probe Hilbert Curve Count Sweep ($C = 1, 2, 3, 4, \\dots$) Benchmark & Plotter.
Uses real empirical data from `evaluation/data/`:
  - `evaluation/data/*_island_*.jsonl` (real evolved 10D aircraft genomes from GA runs)
  - `evaluation/data/real_criterion_benchmarks.json` (Criterion.rs Hilbert encoding latencies)
  - `evaluation/data/real_lead_dht_benchmarks.json` (LEAD DHT SHA-1 vs Hilbert recall baseline)

Evaluates across arbitrary rotated Hilbert curve counts $C \\in \\{1, 2, \\dots, C_{\\max}\\}$:
  1. (a) Spatial $k$-NN Recall@$K$ (Forward & Bidirectional Multi-Probe vs. SHA-1 Hash Baseline)
  2. (b) Top-1 Nearest-Neighbor Hit Rate & Euclidean Distance Approximation Ratio
  3. (c) DHT Index Storage Footprint (KB) & Key Generation Latency ($\\mu$s) from Criterion.rs
  4. (d) Marginal Recall Gain ($\\Delta$Recall / $\\Delta C$) & Candidate Search Cost

Generates:
  - evaluation/figures/fig6_hilbert_curve_sweep.png
  - evaluation/figures/table_fig6_hilbert_curve_sweep.tex
  - evaluation/data/real_hilbert_curve_sweep.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "src"))
from worker.hilbert import BITS, probe_keys

EVAL_DIR = Path(__file__).resolve().parent.parent

# IEEE publication aesthetics matching evaluation/scripts/*
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True,
})

DIMS_10D: List[Tuple[str, float, float]] = [(f"g{i}", 0.0, 1.0) for i in range(10)]


@dataclass(frozen=True)
class CurveSweepMetrics:
    num_curves: int
    total_keys: int
    storage_kb: float
    keygen_latency_us: float
    fwd_recall_mean: float
    fwd_recall_q25: float
    fwd_recall_q75: float
    fwd_top1_hit_pct: float
    fwd_dist_ratio: float
    fwd_unique_candidates: float
    bi_recall_mean: float
    bi_recall_q25: float
    bi_recall_q75: float
    bi_top1_hit_pct: float
    bi_dist_ratio: float
    bi_unique_candidates: float
    sha1_baseline_recall_pct: float


def load_real_genomes_from_data_dir(data_dir: Path, max_samples: int = 250) -> np.ndarray:
    """Loads real 10D evaluated aircraft genomes from evaluation/data/*.jsonl."""
    jsonl_files = sorted(data_dir.glob("*_island_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No *_island_*.jsonl files found in {data_dir}")

    seen = set()
    unique_genomes: List[List[float]] = []

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                record = json.loads(line_str)
                best = record.get("best_genome")
                if best and len(best) == 10:
                    key = tuple(round(float(x), 6) for x in best)
                    if key not in seen:
                        seen.add(key)
                        unique_genomes.append([float(x) for x in best])
                for ind in record.get("population", []):
                    if len(ind) == 10:
                        key = tuple(round(float(x), 6) for x in ind)
                        if key not in seen:
                            seen.add(key)
                            unique_genomes.append([float(x) for x in ind])

    if len(unique_genomes) > max_samples:
        # Deterministic stratified subsample across generational trajectory
        indices = np.linspace(0, len(unique_genomes) - 1, max_samples, dtype=int)
        unique_genomes = [unique_genomes[i] for i in indices]

    return np.asarray(unique_genomes, dtype=np.float64)


def load_benchmark_metadata(data_dir: Path) -> Tuple[dict, dict]:
    """Loads Criterion.rs and LEAD DHT benchmark JSONs from evaluation/data/."""
    crit_path = data_dir / "real_criterion_benchmarks.json"
    dht_path = data_dir / "real_lead_dht_benchmarks.json"

    with open(crit_path, "r", encoding="utf-8") as f:
        crit_data = json.load(f)
    with open(dht_path, "r", encoding="utf-8") as f:
        dht_data = json.load(f)

    return crit_data, dht_data


def evaluate_curve_sweep(
    genomes: np.ndarray,
    crit_data: dict,
    dht_data: dict,
    max_curves: int = 10,
    k_neighbors: int = 10,
    wider_factor: int = 3,
    bits: int = BITS,
) -> List[CurveSweepMetrics]:
    """Sweeps C = 1..max_curves on real 10D genomes from evaluation/data/."""
    n_samples = len(genomes)

    # Pairwise 10D Euclidean distance matrix & Ground-Truth k-NN
    diff = genomes[:, np.newaxis, :] - genomes[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    gt_sets: List[set[int]] = []
    gt_mean_dists: List[float] = []
    gt_top1: List[int] = []

    for i in range(n_samples):
        order = [j for j in np.argsort(dist_matrix[i]) if j != i][:k_neighbors]
        gt_sets.append(set(order))
        gt_mean_dists.append(float(np.mean([dist_matrix[i, j] for j in order])))
        gt_top1.append(int(order[0]))

    # Encode all genomes across max_curves using hilbert_rs / worker.hilbert
    keys_by_sample: List[List[str]] = []
    payloads_by_sample: List[str] = []
    for i, g in enumerate(genomes):
        cfg = {f"g{d}": float(g[d]) for d in range(10)}
        keys_by_sample.append(probe_keys(cfg, DIMS_10D, bits=bits, num_curves=max_curves))
        payloads_by_sample.append(
            json.dumps({"genes": [round(float(x), 6) for x in g], "generation": i})
        )

    # Build per-curve sorted B-Tree tables & compute exact byte size per curve
    sorted_keys_per_curve: List[List[str]] = []
    sorted_ids_per_curve: List[List[int]] = []
    bytes_per_curve: List[int] = []
    btree_node_overhead_bytes = 48  # From plot_rmi_learned_index.py BTreeMap overhead

    for c in range(max_curves):
        pairs = sorted((keys_by_sample[i][c], i) for i in range(n_samples))
        sorted_keys_per_curve.append([k for k, _ in pairs])
        sorted_ids_per_curve.append([idx for _, idx in pairs])
        raw_bytes = sum(
            len(keys_by_sample[i][c].encode("utf-8"))
            + len(payloads_by_sample[i].encode("utf-8"))
            + btree_node_overhead_bytes
            for i in range(n_samples)
        )
        bytes_per_curve.append(raw_bytes)

    # Criterion.rs empirical per-curve encoding latency
    hilbert_crit = crit_data["hilbert_encoding"]
    per_curve_encode_us = hilbert_crit["encode_hex_10d_ns"] / 1000.0
    json_serialize_overhead_us = max(
        0.0,
        hilbert_crit["keys_for_10d_us"] - 3.0 * per_curve_encode_us,
    )

    # SHA-1 hash recall baseline from real_lead_dht_benchmarks.json
    hash_recalls = dht_data["spatial_recall"]["hash_recall"]

    wide_count = k_neighbors * wider_factor + 1
    half_window = (k_neighbors * wider_factor) // 2

    results: List[CurveSweepMetrics] = []

    for c_count in range(1, max_curves + 1):
        fwd_recalls: List[float] = []
        fwd_ratios: List[float] = []
        fwd_cands_list: List[int] = []
        fwd_top1_hits = 0

        bi_recalls: List[float] = []
        bi_ratios: List[float] = []
        bi_cands_list: List[int] = []
        bi_top1_hits = 0

        for i in range(n_samples):
            # Forward multi-probe range scan across curves 0..c_count
            fwd_cands: set[int] = set()
            for c in range(c_count):
                s_keys = sorted_keys_per_curve[c]
                s_ids = sorted_ids_per_curve[c]
                target_key = keys_by_sample[i][c]
                pos = bisect.bisect_left(s_keys, target_key)
                for j in range(pos, min(pos + wide_count, len(s_keys))):
                    if s_ids[j] != i:
                        fwd_cands.add(s_ids[j])

            fwd_ranked = sorted(fwd_cands, key=lambda j: dist_matrix[i, j])[:k_neighbors]
            fwd_top_set = set(fwd_ranked)
            fwd_recalls.append(100.0 * len(gt_sets[i] & fwd_top_set) / k_neighbors)
            if fwd_ranked:
                fwd_avg_d = float(np.mean([dist_matrix[i, j] for j in fwd_ranked]))
                fwd_ratios.append(fwd_avg_d / gt_mean_dists[i])
            fwd_cands_list.append(len(fwd_cands))
            if gt_top1[i] in fwd_top_set:
                fwd_top1_hits += 1

            # Bidirectional multi-probe range scan across curves 0..c_count
            bi_cands: set[int] = set()
            for c in range(c_count):
                s_keys = sorted_keys_per_curve[c]
                s_ids = sorted_ids_per_curve[c]
                target_key = keys_by_sample[i][c]
                pos = bisect.bisect_left(s_keys, target_key)
                lo = max(0, pos - half_window)
                hi = min(len(s_keys), pos + half_window + 1)
                for j in range(lo, hi):
                    if s_ids[j] != i:
                        bi_cands.add(s_ids[j])

            bi_ranked = sorted(bi_cands, key=lambda j: dist_matrix[i, j])[:k_neighbors]
            bi_top_set = set(bi_ranked)
            bi_recalls.append(100.0 * len(gt_sets[i] & bi_top_set) / k_neighbors)
            if bi_ranked:
                bi_avg_d = float(np.mean([dist_matrix[i, j] for j in bi_ranked]))
                bi_ratios.append(bi_avg_d / gt_mean_dists[i])
            bi_cands_list.append(len(bi_cands))
            if gt_top1[i] in bi_top_set:
                bi_top1_hits += 1

        fwd_arr = np.asarray(fwd_recalls, dtype=np.float64)
        bi_arr = np.asarray(bi_recalls, dtype=np.float64)
        sha1_rec = float(hash_recalls[min(c_count - 1, len(hash_recalls) - 1)])

        results.append(
            CurveSweepMetrics(
                num_curves=c_count,
                total_keys=n_samples * c_count,
                storage_kb=sum(bytes_per_curve[:c_count]) / 1024.0,
                keygen_latency_us=json_serialize_overhead_us + c_count * per_curve_encode_us,
                fwd_recall_mean=float(np.mean(fwd_arr)),
                fwd_recall_q25=float(np.percentile(fwd_arr, 25)),
                fwd_recall_q75=float(np.percentile(fwd_arr, 75)),
                fwd_top1_hit_pct=100.0 * fwd_top1_hits / n_samples,
                fwd_dist_ratio=float(np.mean(fwd_ratios)),
                fwd_unique_candidates=float(np.mean(fwd_cands_list)),
                bi_recall_mean=float(np.mean(bi_arr)),
                bi_recall_q25=float(np.percentile(bi_arr, 25)),
                bi_recall_q75=float(np.percentile(bi_arr, 75)),
                bi_top1_hit_pct=100.0 * bi_top1_hits / n_samples,
                bi_dist_ratio=float(np.mean(bi_ratios)),
                bi_unique_candidates=float(np.mean(bi_cands_list)),
                sha1_baseline_recall_pct=sha1_rec,
            )
        )

    return results


def plot_and_export_curve_sweep(
    metrics: List[CurveSweepMetrics],
    n_samples: int,
    k_neighbors: int,
    out_dir: Path,
    data_dir: Path,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save empirical sweep dataset into evaluation/data/
    json_out = data_dir / "real_hilbert_curve_sweep.json"
    with open(json_out, "w", encoding="utf-8") as f_json:
        json.dump(
            {
                "n_samples": n_samples,
                "k_neighbors": k_neighbors,
                "sweep": [asdict(m) for m in metrics],
            },
            f_json,
            indent=2,
        )

    cs = np.array([m.num_curves for m in metrics])
    fwd_rec = np.array([m.fwd_recall_mean for m in metrics])
    fwd_q25 = np.array([m.fwd_recall_q25 for m in metrics])
    fwd_q75 = np.array([m.fwd_recall_q75 for m in metrics])

    bi_rec = np.array([m.bi_recall_mean for m in metrics])
    bi_q25 = np.array([m.bi_recall_q25 for m in metrics])
    bi_q75 = np.array([m.bi_recall_q75 for m in metrics])

    sha1_rec = np.array([m.sha1_baseline_recall_pct for m in metrics])
    fwd_top1 = np.array([m.fwd_top1_hit_pct for m in metrics])
    bi_top1 = np.array([m.bi_top1_hit_pct for m in metrics])
    fwd_ratio = np.array([m.fwd_dist_ratio for m in metrics])
    bi_ratio = np.array([m.bi_dist_ratio for m in metrics])

    storage_kb = np.array([m.storage_kb for m in metrics])
    keygen_us = np.array([m.keygen_latency_us for m in metrics])
    fwd_cands = np.array([m.fwd_unique_candidates for m in metrics])
    bi_cands = np.array([m.bi_unique_candidates for m in metrics])

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6), dpi=300)
    (ax1, ax2), (ax3, ax4) = axes

    # ─────────────────────────────────────────────────────────────
    # Panel (a): k-NN Spatial Recall vs. Number of Curves (C)
    # ─────────────────────────────────────────────────────────────
    ax1.plot(
        cs,
        bi_rec,
        "s-",
        color="#059669",
        linewidth=2.4,
        markersize=6.5,
        zorder=4,
        label="Multi-Probe Hilbert Bidirectional (IQR)",
    )
    ax1.fill_between(cs, bi_q25, bi_q75, color="#10b981", alpha=0.12, zorder=2)

    ax1.plot(
        cs,
        fwd_rec,
        "o-",
        color="#2563eb",
        linewidth=2.4,
        markersize=6.5,
        zorder=4,
        label="Multi-Probe Hilbert Forward (IQR)",
    )
    ax1.fill_between(cs, fwd_q25, fwd_q75, color="#3b82f6", alpha=0.10, zorder=2)

    ax1.plot(
        cs,
        sha1_rec,
        "d--",
        color="#ef4444",
        linewidth=2.0,
        markersize=5.5,
        zorder=3,
        label="Standard Consistent Hash (SHA-1)",
    )
    ax1.axvline(3, color="#dc2626", linestyle=":", linewidth=1.8, zorder=3, label="System Default ($C=3$)")

    # Annotate key recall milestones on Bidirectional curve
    for c_idx, y_off in ((0, 8), (min(2, len(cs) - 1), 8), (len(cs) - 1, -16)):
        ax1.annotate(
            f"{bi_rec[c_idx]:.1f}%",
            xy=(cs[c_idx], bi_rec[c_idx]),
            xytext=(0, y_off),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#065f46",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=0.8),
        )

    ax1.set_xticks(cs)
    ax1.set_ylim(0, 106)
    ax1.set_xlabel("Number of Rotated Hilbert Curves ($C$)")
    ax1.set_ylabel(f"10D Spatial Recall@{k_neighbors} (%)")
    ax1.set_title(f"(a) Spatial Neighbor Recall@{k_neighbors} Scaling ($N={n_samples}$)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.94)

    # ─────────────────────────────────────────────────────────────
    # Panel (b): Top-1 Hit Rate & Approximation Distance Ratio
    # ─────────────────────────────────────────────────────────────
    ax2.plot(cs, bi_top1, "s-", color="#059669", linewidth=2.4, markersize=6.5, label="Top-1 Hit Rate (Bidirectional)")
    ax2.plot(cs, fwd_top1, "o-", color="#2563eb", linewidth=2.4, markersize=6.5, label="Top-1 Hit Rate (Forward)")
    ax2.axvline(3, color="#dc2626", linestyle=":", linewidth=1.8, label="System Default ($C=3$)")
    ax2.set_xticks(cs)
    ax2.set_ylim(30, 104)
    ax2.set_xlabel("Number of Rotated Hilbert Curves ($C$)")
    ax2.set_ylabel("True #1 Nearest-Neighbor Hit Rate (%)", color="#1e3a8a")
    ax2.tick_params(axis="y", labelcolor="#1e3a8a")
    ax2.set_title("(b) Top-1 Recovery & Distance Ratio ($d_{\\mathrm{ret}}/d_{\\mathrm{GT}}$)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        cs,
        fwd_ratio,
        "^--",
        color="#d97706",
        linewidth=2.0,
        markersize=5.5,
        label="Distance Ratio $d_{\\mathrm{ret}}/d_{\\mathrm{GT}}$ (Fwd)",
    )
    ax2_twin.plot(
        cs,
        bi_ratio,
        "v--",
        color="#7c3aed",
        linewidth=2.0,
        markersize=5.5,
        label="Distance Ratio $d_{\\mathrm{ret}}/d_{\\mathrm{GT}}$ (Bi-dir)",
    )
    ax2_twin.set_ylim(1.00, 1.42)
    ax2_twin.set_ylabel("Euclidean Distance Ratio ($1.00\\times = \\mathrm{Optimal}$)", color="#92400e")
    ax2_twin.tick_params(axis="y", labelcolor="#92400e")

    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
    ax2.legend(
        lines_2 + lines_2t,
        labels_2 + labels_2t,
        loc="center right",
        frameon=True,
        framealpha=0.94,
        fontsize=8.5,
    )

    # ─────────────────────────────────────────────────────────────
    # Panel (c): Storage Footprint (KB) & Key Generation Latency (µs)
    # ─────────────────────────────────────────────────────────────
    # Separate vertical bands so bar labels and latency curve never overlap
    bars = ax3.bar(
        cs,
        storage_kb,
        color="#8b5cf6",
        alpha=0.82,
        edgecolor="#2e1065",
        linewidth=0.9,
        width=0.56,
        zorder=3,
        label="B-Tree + Payload Index Footprint (KB)",
    )
    ax3.axvline(3, color="#dc2626", linestyle=":", linewidth=1.8, zorder=4, label="System Default ($C=3$)")
    ax3.set_xticks(cs)
    ax3.set_xlabel("Number of Rotated Hilbert Curves ($C$)")
    ax3.set_ylabel("DHT Storage Footprint (KB)", color="#5b21b6")
    ax3.tick_params(axis="y", labelcolor="#5b21b6")
    ax3.set_ylim(0, max(storage_kb) * 1.55)
    ax3.set_title("(c) Index Space Overhead & Key Encoding Latency", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.35, zorder=1)

    # Clean, readable storage labels placed directly above each bar
    for rect, kb in zip(bars, storage_kb):
        ax3.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 18.0,
            f"{kb:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#4c1d95",
        )

    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        cs,
        keygen_us,
        "o-",
        color="#d97706",
        linewidth=2.4,
        markersize=6.5,
        zorder=5,
        label="Multi-Probe Key Gen Latency ($\\mu$s)",
    )
    ax3_twin.set_ylabel("Key Generation Latency ($\\mu$s)", color="#b45309")
    ax3_twin.tick_params(axis="y", labelcolor="#b45309")
    # Position the latency curve in the upper band above the bar labels
    ax3_twin.set_ylim(-1.8, max(keygen_us) * 1.15)
    ax3_twin.set_yticks([0, 1, 2, 3, 4, 5, 6])

    for c_idx in sorted({0, min(2, len(cs) - 1), min(5, len(cs) - 1), len(cs) - 1}):
        ax3_twin.annotate(
            f"{keygen_us[c_idx]:.2f} $\\mu$s",
            xy=(cs[c_idx], keygen_us[c_idx]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#92400e",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#fffbeb", edgecolor="#fcd34d", lw=0.8),
        )

    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3t, labels_3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(
        lines_3 + lines_3t,
        labels_3 + labels_3t,
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=8.8,
    )

    # ─────────────────────────────────────────────────────────────
    # Panel (d): Marginal Recall Gain (ΔRecall / ΔC) & Search Cost
    # ─────────────────────────────────────────────────────────────
    # Marginal gain relative to previous curve count (for C=1, gain over SHA-1 baseline)
    marginal_bi = np.diff(bi_rec, prepend=sha1_rec[0])
    marginal_fwd = np.diff(fwd_rec, prepend=sha1_rec[0])
    w = 0.36
    b_bi = ax4.bar(
        cs - w / 2,
        marginal_bi,
        width=w,
        color="#059669",
        alpha=0.85,
        edgecolor="#064e3b",
        linewidth=0.8,
        zorder=3,
        label="$\\Delta$Recall@10 Gain per Added Curve (Bi-dir)",
    )
    ax4.bar(
        cs + w / 2,
        marginal_fwd,
        width=w,
        color="#2563eb",
        alpha=0.85,
        edgecolor="#1e3a8a",
        linewidth=0.8,
        zorder=3,
        label="$\\Delta$Recall@10 Gain per Added Curve (Fwd)",
    )
    ax4.axvline(3, color="#dc2626", linestyle=":", linewidth=1.8, zorder=4, label="Pareto Knee ($C=3$)")
    ax4.set_xticks(cs)
    ax4.set_ylim(0, max(np.max(marginal_bi), np.max(marginal_fwd)) * 1.32)
    ax4.set_xlabel("Number of Rotated Hilbert Curves ($C$)")
    ax4.set_ylabel("Marginal Recall Gain $\\Delta\\mathrm{Recall}$ (%)")
    ax4.set_title("(d) Diminishing Returns & Candidate Search Cost", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.35, zorder=1)

    for rect, val in zip(b_bi, marginal_bi):
        if val >= 2.0:
            ax4.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.8,
                f"+{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#065f46",
            )

    ax4_twin = ax4.twinx()
    ax4_twin.plot(
        cs,
        (fwd_cands / n_samples) * 100.0,
        "d--",
        color="#475569",
        linewidth=2.0,
        markersize=5.5,
        zorder=5,
        label="Evaluated Candidates (Fwd, % of $N$)",
    )
    ax4_twin.plot(
        cs,
        (bi_cands / n_samples) * 100.0,
        "s:",
        color="#0f172a",
        linewidth=1.9,
        markersize=5.0,
        zorder=5,
        label="Evaluated Candidates (Bi-dir, % of $N$)",
    )
    ax4_twin.set_ylabel("Deduplicated Candidates Scanned (% of $N$)", color="#334155")
    ax4_twin.tick_params(axis="y", labelcolor="#334155")
    ax4_twin.set_ylim(-14, 74)
    ax4_twin.set_yticks([0, 10, 20, 30, 40, 50, 60])

    lines_4, labels_4 = ax4.get_legend_handles_labels()
    lines_4t, labels_4t = ax4_twin.get_legend_handles_labels()
    ax4.legend(
        lines_4 + lines_4t,
        labels_4 + labels_4t,
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=8.2,
    )

    out_png = out_dir / "fig6_hilbert_curve_sweep.png"
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_png}")

    # Export publication-ready LaTeX table
    tex_path = out_dir / "table_fig6_hilbert_curve_sweep.tex"
    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{Multi-Probe Hilbert Curve Count ($C$) Scaling: Spatial Recall, Storage Overhead, and Latency}
\label{tab:hilbert_curve_sweep}
\begin{tabular}{ccccccccc}
\hline
\textbf{Curves ($C$)} & \textbf{Total Keys} & \textbf{Index Space} & \textbf{KeyGen ($\mu$s)} & \textbf{Fwd Recall@10} & \textbf{Bi-dir Recall@10} & \textbf{Top-1 Hit (Bi)} & \textbf{Dist Ratio} & \textbf{Candidates} \\
\hline
""")
        for m in metrics:
            cand_pct = 100.0 * m.bi_unique_candidates / n_samples
            f_tex.write(
                f"$C = {m.num_curves}$ & {m.total_keys:,} & {m.storage_kb:.1f}~KB & "
                f"{m.keygen_latency_us:.2f}~$\\mu$s & {m.fwd_recall_mean:.1f}\\% & "
                f"{m.bi_recall_mean:.1f}\\% & {m.bi_top1_hit_pct:.1f}\\% & "
                f"{m.bi_dist_ratio:.2f}$\\times$ & {m.bi_unique_candidates:.1f} ({cand_pct:.1f}\\%) \\\\\n"
            )
        f_tex.write(r"""\hline
\end{tabular}
\end{table}
""")
    print(f"Generated: {tex_path}")
    print(f"Saved empirical data: {json_out}")
    return out_png, tex_path, json_out


# -----------------------------------------------------------------------------
# Pytest test entry point
# -----------------------------------------------------------------------------

def test_curve_sweep_uses_evaluation_data_and_plots(tmp_path: Path) -> None:
    data_dir = EVAL_DIR / "data"
    genomes = load_real_genomes_from_data_dir(data_dir, max_samples=120)
    crit_data, dht_data = load_benchmark_metadata(data_dir)

    assert genomes.shape == (120, 10)
    metrics = evaluate_curve_sweep(genomes, crit_data, dht_data, max_curves=8, k_neighbors=10)
    assert len(metrics) == 8

    # Verify linear storage scaling and monotonic recall growth
    unit_kb = metrics[0].storage_kb
    for prev, curr in zip(metrics, metrics[1:]):
        assert curr.total_keys == 120 * curr.num_curves
        assert abs(curr.storage_kb - unit_kb * curr.num_curves) < 1.0
        assert curr.fwd_recall_mean >= prev.fwd_recall_mean - 1e-9
        assert curr.bi_recall_mean >= prev.bi_recall_mean - 1e-9
        assert curr.bi_dist_ratio <= prev.bi_dist_ratio + 1e-9

    # Multi-probe significantly outperforms single curve and SHA-1 baseline
    assert metrics[2].bi_recall_mean > metrics[0].bi_recall_mean + 20.0
    assert metrics[2].bi_recall_mean > metrics[2].sha1_baseline_recall_pct

    png_path, tex_path, json_path = plot_and_export_curve_sweep(
        metrics,
        n_samples=len(genomes),
        k_neighbors=10,
        out_dir=tmp_path,
        data_dir=tmp_path,
    )
    assert png_path.is_file() and png_path.stat().st_size > 15_000
    assert tex_path.is_file() and tex_path.stat().st_size > 200
    assert json_path.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Probe Hilbert Curve Count Sweep Plotter")
    parser.add_argument("--data-dir", type=str, default=str(EVAL_DIR / "data"))
    parser.add_argument("--out-dir", type=str, default=str(EVAL_DIR / "figures"))
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--max-curves", type=int, default=10)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--wider-factor", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    print(f"Loading real 10D aircraft genomes from {data_dir}...")
    genomes = load_real_genomes_from_data_dir(data_dir, max_samples=args.samples)
    crit_data, dht_data = load_benchmark_metadata(data_dir)
    print(f"Evaluating curve sweep C=1..{args.max_curves} across {len(genomes)} empirical genomes...")

    metrics = evaluate_curve_sweep(
        genomes,
        crit_data,
        dht_data,
        max_curves=args.max_curves,
        k_neighbors=args.k_neighbors,
        wider_factor=args.wider_factor,
    )
    plot_and_export_curve_sweep(
        metrics,
        n_samples=len(genomes),
        k_neighbors=args.k_neighbors,
        out_dir=out_dir,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()
