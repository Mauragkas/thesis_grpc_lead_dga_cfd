#!/usr/bin/env python3
"""
Genetic Algorithm Evolutionary Operators Microbenchmark Plotter & Table Generator.
Visualizes real Criterion benchmark results for:
  - Random population generation scaling (N in {50, 200, 1000})
  - (μ + λ) survivor selection & Pareto ranking (N in {100, 500, 2000})
  - Gaussian offspring mutation and hypercube clipping loop

Generates:
  - evaluation/figures/fig6_ga_operators_benchmark.png
  - evaluation/figures/table3_ga_operators_and_gene_store.tex
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


def plot_ga_operators(bench_path: Path, out_dir: Path):
    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ga = data["ga_operators"]
    gene_store = data["gene_store"]

    rand_sizes = np.array(ga["random_population_pop_sizes"])
    rand_us = np.array(ga["random_population_us"])

    sel_sizes = np.array(ga["select_survivors_pop_sizes"])
    sel_us = np.array(ga["select_survivors_us"])

    mut_100_us = ga["mutate_and_clip_100_children_us"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

    # 1. Left Panel: Operator Latency vs. Population Size
    ax1.plot(rand_sizes, rand_us, marker="o", color="#2563eb", linewidth=2.0, label="Random Pop. Init ($d=10$)")
    ax1.plot(sel_sizes, sel_us, marker="s", color="#8b5cf6", linewidth=2.0, label="Survivor Selection (Sort + Top-50%)")

    # Linear extrapolation / throughput annotation
    ax1.axhline(y=mut_100_us, color="#10b981", linestyle="--", linewidth=1.8, label=f"Mutate & Clip 100 Children ({mut_100_us:.2f} µs)")

    ax1.set_xlabel("Population Size ($P$)")
    ax1.set_ylabel("Execution Latency (µs)")
    ax1.set_title("(a) GA Evolutionary Operator Scaling", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    # 2. Right Panel: Operator Throughput (Individuals Processed per Millisecond)
    rand_tp = (rand_sizes / rand_us) * 1000.0  # indiv / ms
    sel_tp = (sel_sizes / sel_us) * 1000.0
    mut_tp = (100.0 / mut_100_us) * 1000.0

    labels = ["Random Pop.\n(P=1,000)", "Selection\n(P=2,000)", "Mutation\n(100 Children)"]
    tp_values = [rand_tp[-1] / 1000.0, sel_tp[-1] / 1000.0, mut_tp / 1000.0]  # Mega-individuals / sec
    colors = ["#2563eb", "#8b5cf6", "#10b981"]

    bars = ax2.bar(labels, tp_values, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.2, width=0.55)
    for bar, val in zip(bars, tp_values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{val:.1f} M/s", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax2.set_ylabel("Throughput ($10^6$ Individuals / sec)")
    ax2.set_title("(b) Operator Execution Bandwidth", fontweight="bold")
    ax2.set_ylim(0, max(tp_values) * 1.25)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "fig6_ga_operators_benchmark.png"
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_png}")

    # Generate LaTeX table combining GA operators and Gene Store microbenchmarks
    tex_path = out_dir / "table3_ga_operators_and_gene_store.tex"
    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{In-Memory Genetic Algorithm Operator and Gene Store Microbenchmarks}
\label{tab:ga_operators_gene_store}
\begin{tabular}{llr}
\hline
\textbf{Subsystem} & \textbf{Microbenchmark Operation} & \textbf{Measured Mean Latency} \\
\hline
\textbf{GA Operators} & Random Population Sampling ($P=50$) & """ + f"{rand_us[0]:.2f}" + r"""~$\mu\text{s}$ \\
& Random Population Sampling ($P=200$) & """ + f"{rand_us[1]:.2f}" + r"""~$\mu\text{s}$ \\
& Random Population Sampling ($P=1{,}000$) & """ + f"{rand_us[2]:.2f}" + r"""~$\mu\text{s}$ \\
& $(\mu + \lambda)$ Survivor Selection ($P=100$) & """ + f"{sel_us[0]:.2f}" + r"""~$\mu\text{s}$ \\
& $(\mu + \lambda)$ Survivor Selection ($P=500$) & """ + f"{sel_us[1]:.2f}" + r"""~$\mu\text{s}$ \\
& $(\mu + \lambda)$ Survivor Selection ($P=2{,}000$) & """ + f"{sel_us[2]:.2f}" + r"""~$\mu\text{s}$ \\
& Gaussian Mutation \& Clamping (100 Children) & """ + f"{mut_100_us:.2f}" + r"""~$\mu\text{s}$ \\
\hline
\textbf{Gene Store} & 10D Vectorized Euclidean Distance & """ + f"{gene_store['euclidean_10d_ns']:.2f}" + r"""~$\text{ns}$ \\
& Exact Cache Hit ($N=500$) & """ + f"{gene_store['exact_lookup_hit_ns']:.2f}" + r"""~$\text{ns}$ \\
& Exact Cache Miss ($N=500$) & """ + f"{gene_store['exact_lookup_miss_ns']:.2f}" + r"""~$\text{ns}$ \\
& $k$-NN Spatial Query ($N=100, k=10$) & """ + f"{gene_store['knn_us'][0]:.2f}" + r"""~$\mu\text{s}$ \\
& $k$-NN Spatial Query ($N=1{,}000, k=10$) & """ + f"{gene_store['knn_us'][1]:.2f}" + r"""~$\mu\text{s}$ \\
& $k$-NN Spatial Query ($N=5{,}000, k=10$) & """ + f"{gene_store['knn_us'][2]:.2f}" + r"""~$\mu\text{s}$ \\
& Generation TTL Eviction Scan ($N=1{,}000$) & """ + f"{gene_store['evict_expired_1000_records_us']:.2f}" + r"""~$\mu\text{s}$ \\
\hline
\end{tabular}
\end{table}
""")
    print(f"Generated: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot GA Operators Microbenchmarks")
    parser.add_argument("--benchmarks", type=str, default="evaluation/data/real_criterion_benchmarks.json")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_ga_operators(Path(args.benchmarks), Path(args.out_dir))


if __name__ == "__main__":
    main()
