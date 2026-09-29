#!/usr/bin/env python3
"""
PCA Evolution & Fitness Manifold Projection for Thesis Paper.
Reduces the 10-dimensional genome search space to a 2D principal component
manifold (PC1 vs. PC2) to visualize how the GA population migrates and contracts
over generational time.

Visual encodings:
  - Coordinate position (X, Y): 2D PCA projection of the 10D genome.
  - Marker Opacity / Alpha: Generation index (early gens faint, later gens opaque).
  - Colormap / Heatmap: Fitness score (penalties clipped, highlighting high-fitness convergence).
  - Trajectory Line & Centroids: Mean population migration path across epochs.

Generates:
  - figures/fig1_pca_evolution_manifold.png
  - figures/table_fig1_pca_loadings.tex
"""

import argparse
import glob
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.decomposition import PCA

# Publication aesthetics
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

FEATURE_NAMES = [
    "wing_span",
    "wing_root_chord",
    "wing_tip_chord",
    "wing_sweep",
    "wing_dihedral",
    "wing_washout",
    "airfoil_camber",
    "airfoil_thickness",
    "fuselage_diameter",
    "fuselage_length",
]


def load_population_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[float]]:
    all_genomes = []
    all_gens = []
    best_trajectory = []
    best_fitnesses = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gen = r["generation"]
            best_trajectory.append(np.array(r["best_genome"]))
            best_fitnesses.append(r["best_fitness"])

            pop = r.get("population", [])
            for ind in pop:
                all_genomes.append(ind)
                all_gens.append(gen)

    return (
        np.array(all_genomes),
        np.array(all_gens),
        best_trajectory,
        best_fitnesses,
    )


def plot_pca_manifold(jsonl_path: Path, out_dir: Path):
    print(f"Loading population from {jsonl_path}...")
    all_genomes, all_gens, best_trajectory, best_fitnesses = load_population_data(jsonl_path)

    if len(all_genomes) == 0:
        print(f"No population data found in {jsonl_path}.")
        return

    # Fit PCA across all evaluated individuals
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(all_genomes)
    exp_var = pca.explained_variance_ratio_ * 100.0

    max_gen = np.max(all_gens)

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    key_gens = sorted(list(set([1, 5, 10, 20, 35, int(max_gen)])))
    
    for g in key_gens:
        mask = (all_gens == g)
        if not np.any(mask):
            continue
        alpha = 0.15 + 0.75 * (g / max_gen)
        size = 18 + 20 * (g / max_gen)
        
        label = f"Gen {g}" if g in [1, 10, int(max_gen)] else None
        scatter = ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[g] * np.sum(mask),
            cmap="plasma",
            vmin=1,
            vmax=max_gen,
            alpha=alpha,
            s=size,
            edgecolors="none" if g < max_gen else "black",
            linewidths=0.5 if g == max_gen else 0,
            label=label,
            zorder=2 + g,
        )

    # 2. Project Best Ever Trajectory
    best_pca = pca.transform(np.array(best_trajectory))
    ax.plot(
        best_pca[:, 0],
        best_pca[:, 1],
        color="#0f172a",
        linestyle="-",
        linewidth=2.2,
        alpha=0.85,
        label="Optimal Search Trajectory",
        zorder=100,
    )

    ax.scatter(
        best_pca[0, 0],
        best_pca[0, 1],
        marker="s",
        color="#38bdf8",
        s=120,
        edgecolors="black",
        linewidths=1.2,
        label="Initial Best (Gen 1)",
        zorder=101,
    )
    ax.scatter(
        best_pca[-1, 0],
        best_pca[-1, 1],
        marker="*",
        color="#e11d48",
        s=260,
        edgecolors="black",
        linewidths=1.2,
        label=f"Evolved Optimum ($F={best_fitnesses[-1]:.2f}$)",
        zorder=102,
    )

    ax.set_xlabel(f"Principal Component 1 ({exp_var[0]:.1f}% variance)")
    ax.set_ylabel(f"Principal Component 2 ({exp_var[1]:.1f}% variance)")
    ax.set_title("2D PCA Manifold Projection of 10-Dimensional Population Evolution")
    ax.grid(True, linestyle="--", alpha=0.35)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Generational Progression (Opacity & Heatmap)")

    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig1_pca_evolution_manifold.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

    # Export LaTeX table for PCA loadings
    tex_path = out_dir / "table_fig1_pca_loadings.tex"
    loadings = pca.components_
    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{Principal Component Loadings and Explained Variance of 10D Genome Space}
\label{tab:pca_loadings}
\begin{tabular}{lcc}
\hline
\textbf{Genome Parameter ($x_i$)} & \textbf{PC1 Loading (""" + f"{exp_var[0]:.1f}\\%" + r""")} & \textbf{PC2 Loading (""" + f"{exp_var[1]:.1f}\\%" + r""")} \\
\hline
""")
        for idx, feat in enumerate(FEATURE_NAMES):
            feat_clean = feat.replace("_", r"\_")
            pc1_l = loadings[0, idx]
            pc2_l = loadings[1, idx]
            f_tex.write(f"\\texttt{{{feat_clean}}} & {pc1_l:+.3f} & {pc2_l:+.3f} \\\\\n")
        f_tex.write(r"""\hline
\textbf{Cumulative Variance} & \multicolumn{2}{c}{""" + f"{exp_var[0] + exp_var[1]:.1f}\\%" + r"""} \\
\hline
\end{tabular}
\end{table}
""")
    print(f"Generated: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PCA Evolution Manifold")
    parser.add_argument("--jsonl", type=str, default="", help="Path to JSONL run file")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl) if args.jsonl else None
    if not jsonl_path or not jsonl_path.exists():
        found = sorted(glob.glob("evaluation/data/*_island_*.jsonl"))
        if found:
            jsonl_path = Path(found[0])

    if not jsonl_path or not jsonl_path.exists():
        print("No JSONL file found for PCA analysis.")
        return

    plot_pca_manifold(jsonl_path, Path(args.out_dir))


if __name__ == "__main__":
    main()
