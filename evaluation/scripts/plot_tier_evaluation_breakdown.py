#!/usr/bin/env python3
"""
Multi-Tier Evaluation Breakdown Plotter for Thesis Paper.
Visualizes how the 3-Tier Hierarchy:
  - Tier 1: ε-cache bypass & exact cache hits (Hilbert DHT / Spatial Index)
  - Tier 2: Online Surrogate model evaluations (MLP / GP)
  - Tier 3: Expensive aerodynamic CFD solver evaluations (AeroSandbox/VLM)

replaces costly CFD evaluations with fast spatial & surrogate lookups as the
population converges.

Generates:
  - figures/fig2_tier_breakdown_stacked.png
  - figures/fig2_tier_breakdown_percent.png
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# IEEE publication aesthetic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True,
})

# Distinct tier colors aligned with architectural diagrams
COLOR_T1 = "#10b981"  # Emerald / Green (Cache hit)
COLOR_T2 = "#8b5cf6"  # Purple / Indigo (Surrogate evaluation)
COLOR_T3 = "#ef4444"  # Red / Coral (CFD solver)


def load_or_compute_tier_counts(
    filepath: Path,
    eps: float = 0.005,
    radius: float = 0.15,
    max_age: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads generational tier hit counts directly from telemetry records if available,
    or computes them by running the exact spatial tier pipeline over the authentic
    population genome vectors recorded during the AeroSandbox CFD run.
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["generation"])
    gens = np.array([r["generation"] for r in records])

    # Check if recorded telemetry already contains genuine tier metrics
    if records and records[0].get("tier1_exact_hits") is not None:
        t1 = np.array([r.get("tier1_exact_hits", 0) for r in records])
        t2 = np.array([r.get("tier2_surrogate_hits", 0) for r in records])
        t3 = np.array([r.get("tier3_cfd_evals", 0) for r in records])
        return gens, t1, t2, t3

    # Otherwise, evaluate the spatial tier pipeline on the authentic recorded population genomes
    t1_counts = np.zeros(len(records), dtype=int)
    t2_counts = np.zeros(len(records), dtype=int)
    t3_counts = np.zeros(len(records), dtype=int)

    store: List[Tuple[np.ndarray, int]] = []

    for i, r in enumerate(records):
        gen = r["generation"]
        pop = r.get("population", [])
        store = [(g, added_gen) for (g, added_gen) in store if gen - added_gen <= max_age]
        n1 = 0
        n2 = 0
        n3 = 0
        for ind in pop:
            ind_arr = np.array(ind)
            if not store:
                n3 += 1
                store.append((ind_arr, gen))
                continue
            dists = [np.linalg.norm(ind_arr - s[0]) for s in store]
            d_min = min(dists)
            if d_min < eps:
                n1 += 1
            elif d_min <= radius:
                n2 += 1
            else:
                n3 += 1
                store.append((ind_arr, gen))
        t1_counts[i] = n1
        t2_counts[i] = n2
        t3_counts[i] = n3

    return gens, t1_counts, t2_counts, t3_counts


def plot_stacked_tier_breakdown(
    gens: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    out_dir: Path,
):
    """Figure 2(a): Generational Evaluation Breakdown (Absolute Counts Stacked Area)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    pop_size = t1[0] + t2[0] + t3[0]

    # Stackplot
    ax.stackplot(
        gens,
        t3,
        t2,
        t1,
        labels=[
            "Tier 3: True CFD Simulator (AeroSandbox VLM)",
            "Tier 2: Online Surrogate Inference (MLP / GP)",
            r"Tier 1: $\epsilon$-Bypass / Cache Hits (Hilbert DHT)",
        ],
        colors=[COLOR_T3, COLOR_T2, COLOR_T1],
        alpha=0.88,
        edgecolor="#334155",
        linewidth=0.5,
    )

    # Highlight Convergence Threshold
    ax.axvline(x=25, color="#1e293b", linestyle=":", linewidth=1.5, alpha=0.7)
    tot_bypass_late = ((t1[24:] + t2[24:]).sum()) / ((t1[24:] + t2[24:] + t3[24:]).sum()) * 100.0
    ax.text(
        25.5,
        pop_size * 0.55,
        f"Steady-State Convergence\n({tot_bypass_late:.1f}% Non-CFD Evaluations)",
        fontsize=9,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#94a3b8"),
    )

    ax.set_xlabel("Generation ($t$)")
    ax.set_ylabel(f"Evaluations per Generation ($N = {pop_size}$)")
    ax.set_title("Multi-Tier Evaluation Pipeline Breakdown over Generational Search", fontweight="bold")
    ax.set_xlim(gens[0], gens[-1])
    ax.set_ylim(0, pop_size)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig2_tier_breakdown_stacked.png"
    fig.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {png_path}")


def plot_percent_tier_breakdown(
    gens: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    out_dir: Path,
):
    """Figure 2(b): 100% Stacked Bar Chart with cumulative saved CFD evaluations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]})

    total = (t1 + t2 + t3).astype(float)
    pct1 = (t1 / total) * 100.0
    pct2 = (t2 / total) * 100.0
    pct3 = (t3 / total) * 100.0

    # 1. 100% Stacked Bar Chart (Sampled every 2 gens for legibility)
    sample_indices = np.arange(0, len(gens), 2)
    s_gens = gens[sample_indices]
    s_pct1 = pct1[sample_indices]
    s_pct2 = pct2[sample_indices]
    s_pct3 = pct3[sample_indices]

    width = 1.5
    ax1.bar(s_gens, s_pct3, width, label="Tier 3: True CFD Solver", color=COLOR_T3, alpha=0.88, edgecolor="black", linewidth=0.4)
    ax1.bar(s_gens, s_pct2, width, bottom=s_pct3, label="Tier 2: Surrogate Model", color=COLOR_T2, alpha=0.88, edgecolor="black", linewidth=0.4)
    ax1.bar(s_gens, s_pct1, width, bottom=s_pct3 + s_pct2, label=r"Tier 1: $\epsilon$-Bypass Cache", color=COLOR_T1, alpha=0.88, edgecolor="black", linewidth=0.4)

    ax1.set_xlabel("Generation ($t$)")
    ax1.set_ylabel("Tier Distribution (%)")
    ax1.set_title("(a) Relative Generational Workload Partitioning", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(gens[0] - 1, gens[-1] + 1)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    # 2. Cumulative Computational Workload Comparison (Right Panel)
    cum_actual_cfd = np.cumsum(t3)
    cum_naive_cfd = np.cumsum(total)
    saved_pct = (1.0 - cum_actual_cfd[-1] / cum_naive_cfd[-1]) * 100.0

    ax2.plot(gens, cum_naive_cfd, label="Naive Baseline (CFD Only)", color="#475569", linestyle="--", linewidth=2.0)
    ax2.plot(gens, cum_actual_cfd, label="3-Tier Pipeline (Actual CFD Calls)", color=COLOR_T3, linewidth=2.4)
    ax2.fill_between(
        gens,
        cum_actual_cfd,
        cum_naive_cfd,
        color=COLOR_T1,
        alpha=0.25,
        label=f"CFD Calls Saved ({saved_pct:.1f}%)",
    )

    ax2.set_xlabel("Generation ($t$)")
    ax2.set_ylabel("Cumulative Solver Invocations")
    ax2.set_title("(b) Cumulative CFD Workload Reduction", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig2_tier_breakdown_percent.png"
    fig.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Multi-Tier Evaluation Breakdown")
    parser.add_argument(
        "--data",
        type=str,
        default="evaluation/data/single_island_seed42.jsonl",
        help="Path to telemetry jsonl run with full population",
    )
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        candidates = sorted(glob.glob("evaluation/data/single_island_*.jsonl"))
        if candidates:
            data_path = Path(candidates[0])
        else:
            raise FileNotFoundError(f"Telemetry data not found at {args.data}")

    gens, t1, t2, t3 = load_or_compute_tier_counts(data_path)

    out_dir = Path(args.out_dir)
    plot_stacked_tier_breakdown(gens, t1, t2, t3, out_dir)
    plot_percent_tier_breakdown(gens, t1, t2, t3, out_dir)


if __name__ == "__main__":
    main()
