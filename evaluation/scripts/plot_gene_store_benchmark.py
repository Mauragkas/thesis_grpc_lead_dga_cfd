#!/usr/bin/env python3
"""
In-Memory Gene Store & Spatial Cache Microbenchmark Plotter.
Visualizes real Criterion benchmark results for:
  - In-memory k-NN search scaling over N in {100, 1000, 5000} records
  - Exact cache hit vs. cache miss retrieval latency
  - SIMD Euclidean distance calculation baseline
  - Generation TTL eviction scan latency

Generates:
  - evaluation/figures/fig6_gene_store_benchmark.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# IEEE publication aesthetics
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


def plot_gene_store(bench_path: Path, out_dir: Path):
    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gs = data["gene_store"]

    knn_sizes = np.array(gs["knn_sizes"])
    knn_us = np.array(gs["knn_us"])

    hit_ns = gs["exact_lookup_hit_ns"]
    miss_ns = gs["exact_lookup_miss_ns"]
    euc_ns = gs["euclidean_10d_ns"]
    evict_us = gs["evict_expired_1000_records_us"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

    # 1. Left Panel: k-NN Query Scaling vs. Store Size
    ax1.plot(knn_sizes, knn_us, marker="o", color="#0284c7", linewidth=2.2, label="$k$-NN Search ($k=10, d=10$)")

    # Linear O(N) fit line
    slope, intercept = np.polyfit(knn_sizes, knn_us, 1)
    fit_x = np.linspace(0, 5200, 100)
    ax1.plot(fit_x, slope * fit_x + intercept, linestyle="--", color="#64748b", alpha=0.8,
             label=f"Linear Scan Trend ($O(N \\cdot d)$, {slope*1000:.1f} ns/item)")

    ax1.set_xlabel("Gene Store Cache Size ($N$ records)")
    ax1.set_ylabel("Query Latency (µs)")
    ax1.set_title("(a) In-Memory $k$-NN Spatial Search Scaling", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    # Add callout for 5,000 records
    ax1.annotate(f"{knn_us[-1]:.1f} µs\n(5,000 items)",
                 xy=(knn_sizes[-1], knn_us[-1]), xytext=(knn_sizes[-1] - 1200, knn_us[-1] - 8),
                 arrowprops=dict(facecolor="#0284c7", shrink=0.08, width=1, headwidth=6),
                 fontweight="bold", fontsize=9.5)

    # 2. Right Panel: Cache Lookup & SIMD Distance Metrics
    categories = ["SIMD Dist.\n(10D)", "Exact Hit\n(N=500)", "Exact Miss\n(N=500)", "Evict Scan\n(1k items)"]
    # Convert all to nanoseconds for consistent comparison or use two groups
    times_ns = [euc_ns, hit_ns, miss_ns, evict_us * 1000.0]
    colors = ["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6"]

    bars = ax2.bar(categories, times_ns, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.2, width=0.55)
    ax2.set_yscale("log")
    ax2.set_ylabel("Operation Latency (ns, log-scale)")
    ax2.set_title("(b) Spatial Cache & Distance Operations", fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Annotate bar values
    for bar, val in zip(bars, times_ns):
        if val < 1000:
            txt = f"{val:.1f} ns"
        else:
            txt = f"{val/1000.0:.1f} µs"
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val * 1.35, txt, ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    ax2.set_ylim(1, 100_000)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "fig6_gene_store_benchmark.png"
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot Gene Store Microbenchmarks")
    parser.add_argument("--benchmarks", type=str, default="evaluation/data/real_criterion_benchmarks.json")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_gene_store(Path(args.benchmarks), Path(args.out_dir))


if __name__ == "__main__":
    main()
