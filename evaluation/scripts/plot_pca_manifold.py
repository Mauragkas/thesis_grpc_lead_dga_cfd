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


def load_population_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[float]]:
    """
    Extracts all individuals across generations from a JSONL run file.
    Returns:
      all_genomes: np.ndarray shape (total_individuals, 10)
      all_gens: np.ndarray shape (total_individuals,)
      best_trajectory: List of best genome per generation
      best_fitnesses: List of best fitness per generation
    """
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

    # 1. Plot generational population scatter with temporal opacity
    # Subsample generations to keep visual clarity (e.g. 5 epoch snapshots)
    key_gens = sorted(list(set([1, 5, 10, 20, 35, int(max_gen)])))
    
    # We use a viridis colormap based on generation index, with alpha growing by generation
    for g in key_gens:
        mask = (all_gens == g)
        if not np.any(mask):
            continue
        # Opacity scales from 0.15 (gen 1) to 0.85 (final gen)
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

    # Mark Start (Gen 1) and Optimum (Final Gen)
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

    # Aesthetics
    ax.set_xlabel(f"Principal Component 1 ({exp_var[0]:.1f}% variance)")
    ax.set_ylabel(f"Principal Component 2 ({exp_var[1]:.1f}% variance)")
    ax.set_title("2D PCA Manifold Projection of 10-Dimensional Population Evolution")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Colorbar for generational progression
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Generational Progression (Opacity & Heatmap)")

    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig1_pca_evolution_manifold.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


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
