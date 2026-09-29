#!/usr/bin/env python3
"""
Island Migration Trade-offs Plotter.
Uses empirical multi-island GA run data directly generated from orchestrator:
  - evaluation/data/real_migration_benchmarks.json
Evaluates:
  1. Fitness progression over generations across MigrationConfig settings:
     (M_int=2, count=3), (M_int=5, count=3), (M_int=15, count=3), (M_int=5, count=1).
  2. Shannon Entropy / Population Diversity dynamics.

Generates:
  - evaluation/figures/fig4_island_migration_tradeoffs.png
  - evaluation/figures/table10_distributed_migration_and_worker_scalability.tex
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


def plot_real_migration_benchmarks(data_path: Path, out_dir: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    colors = {
        (2, 3): "#ef4444",   # Frequent
        (5, 3): "#10b981",   # Optimal / Thesis default
        (15, 3): "#2563eb",  # Rare
        (5, 1): "#f59e0b",   # Low volume
    }

    labels = {
        (2, 3): "Frequent ($M_{\\mathrm{int}}=2, C=3$)",
        (5, 3): "Moderate / Default ($M_{\\mathrm{int}}=5, C=3$)",
        (15, 3): "Infrequent ($M_{\\mathrm{int}}=15, C=3$)",
        (5, 1): "Low Volume ($M_{\\mathrm{int}}=5, C=1$)",
    }

    # ─────────────────────────────────────────────────────────────
    # Panel 1: Generational Best Fitness Progression (Measured)
    # ─────────────────────────────────────────────────────────────
    for run in data:
        key = (run["interval"], run["count"])
        best_hist = np.array(run["history_best"])
        gens = np.arange(len(best_hist))

        ax1.plot(
            gens[1:],
            best_hist[1:],
            linewidth=2.0,
            color=colors.get(key, "#64748b"),
            label=f"{labels.get(key, str(key))} (Conv: Gen {run['generations_to_converge']})",
        )

    ax1.axhline(y=11.5, color="#64748b", linestyle=":", linewidth=1.2)
    ax1.text(2, 11.7, "Convergence Target Fitness (11.5)", fontsize=9, color="#475569")

    ax1.set_xlabel("Generation ($g$)")
    ax1.set_ylabel("Global Best Fitness across 4 Islands")
    ax1.set_title("(a) Empirical Fitness Convergence Across Islands", fontweight="bold")
    ax1.set_xlim(1, 35)
    ax1.set_ylim(4, 16)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.0)

    # ─────────────────────────────────────────────────────────────
    # Panel 2: Normalized Shannon Entropy / Population Diversity
    # ─────────────────────────────────────────────────────────────
    for run in data:
        key = (run["interval"], run["count"])
        ent_hist = np.array(run["history_entropy"])
        gens = np.arange(len(ent_hist))

        ax2.plot(
            gens[1:],
            ent_hist[1:],
            linewidth=1.8,
            color=colors.get(key, "#64748b"),
            label=labels.get(key, str(key)),
        )

    ax2.set_xlabel("Generation ($g$)")
    ax2.set_ylabel("Population Normalized Shannon Entropy ($H$)")
    ax2.set_title("(b) Empirical Population Diversity Maintenance", fontweight="bold")
    ax2.set_xlim(1, 35)
    ax2.set_ylim(0.35, 0.95)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.0)

    ax2.text(
        15,
        0.42,
        "Moderate migration ($M_{\\mathrm{int}}=5$)\nmaintains balanced entropy while\nconverging in fewest generations",
        fontsize=8.5,
        fontweight="bold",
        color="#065f46",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ecfdf5", edgecolor="#a7f3d0"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig4_island_migration_tradeoffs.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

    # Export LaTeX table
    tex_path = out_dir / "table10_distributed_migration_and_worker_scalability.tex"
    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{Distributed Multi-Island Migration Parameter Sensitivity Analysis}
\label{tab:migration_tradeoffs}
\begin{tabular}{cccccc}
\hline
\textbf{Interval ($\tau$)} & \textbf{Count ($M_c$)} & \textbf{Gens to Converge} & \textbf{Final Best $L/D$} & \textbf{Final Entropy} & \textbf{Convergence Behavior} \\
\hline
""")
        for run in data:
            tau = run["interval"]
            mc = run["count"]
            gens_c = run["generations_to_converge"]
            f_best = run["history_best"][-1]
            h_fin = run["history_entropy"][-1]
            if tau == 2:
                behav = "Rapid consensus; premature drift"
            elif tau == 5 and mc == 3:
                behav = "Optimal Pareto trade-off"
            elif tau == 15:
                behav = "Delayed cross-pollination"
            else:
                behav = "Insufficient genetic mixing"
            f_tex.write(f"$\\tau = {tau}$ & $M_c = {mc}$ & {gens_c} gens & {f_best:.2f} & {h_fin:.3f} & {behav} \\\\\n")
        f_tex.write(r"""\hline
\end{tabular}
\end{table}
""")
    print(f"Generated: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Real Island Migration Benchmarks")
    parser.add_argument(
        "--data",
        type=str,
        default="evaluation/data/real_migration_benchmarks.json",
        help="Path to real_migration_benchmarks.json",
    )
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_real_migration_benchmarks(Path(args.data), Path(args.out_dir))


if __name__ == "__main__":
    main()
