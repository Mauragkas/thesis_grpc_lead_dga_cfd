#!/usr/bin/env python3
"""
10D Hilbert Spatial Locality Heatmap & Validation Plotter.
Directly uses `worker.hilbert.probe_keys` (or `hilbert_rs` Rust extension) to evaluate:
  1. Correlation between Euclidean Distance in normalized 10D gene space (||x_i - x_j||)
     and 1D distance along the 64-bit Chord DHT ring (|k_i - k_j|).
  2. Single-curve locality vs. 3-curve multi-probe minimum distance.

Generates:
  - evaluation/figures/fig5_hilbert_spatial_heatmap.png
  - evaluation/figures/table_fig5_hilbert_spatial_clustering.tex
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "src"))
from worker.hilbert import probe_keys

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


def parse_key_to_u64(key_str: str) -> int:
    prefix = key_str.split("|")[0]
    hex_prefix = prefix[1:17]  # extract 64-bit prefix
    return int(hex_prefix, 16)


def generate_spatial_locality_heatmap(out_dir: Path, n_samples: int = 250):
    print(f"Sampling {n_samples} points in 10D gene space and computing Hilbert ring keys...")
    rng = np.random.default_rng(42)

    # 10D normalized gene space
    dims = [(f"g{i}", 0.0, 1.0) for i in range(10)]
    points = rng.uniform(0.0, 1.0, size=(n_samples, 10))

    # Compute keys for all points
    single_curve_keys = []
    multi_probe_keys = []

    for pt in points:
        cfg = {f"g{i}": float(pt[i]) for i in range(10)}
        keys = probe_keys(cfg, dims, bits=16)
        u64_keys = [parse_key_to_u64(k) for k in keys]
        single_curve_keys.append(u64_keys[0])
        multi_probe_keys.append(u64_keys)

    single_curve_keys = np.array(single_curve_keys)

    # Compute pairwise Euclidean distances and 1D Ring distances
    eucl_dists = []
    single_ring_dists = []
    multi_ring_dists = []

    max_u64 = 2**64 - 1

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # 10D Euclidean distance
            d_spatial = float(np.linalg.norm(points[i] - points[j]))
            eucl_dists.append(d_spatial)

            # Single-curve 1D ring distance (normalized [0, 1])
            k1 = single_curve_keys[i]
            k2 = single_curve_keys[j]
            d_ring_single = abs(k1 - k2) / max_u64
            single_ring_dists.append(d_ring_single)

            # Multi-probe (3 curves): min distance across rotated curves
            d_multi_min = min(
                abs(multi_probe_keys[i][c] - multi_probe_keys[j][c]) / max_u64
                for c in range(3)
            )
            multi_ring_dists.append(d_multi_min)

    eucl_dists = np.array(eucl_dists)
    single_ring_dists = np.array(single_ring_dists)
    multi_ring_dists = np.array(multi_ring_dists)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    # ─────────────────────────────────────────────────────────────
    # Panel 1: Single-Curve 10D Hilbert 2D Density Heatmap
    # ─────────────────────────────────────────────────────────────
    h1 = ax1.hexbin(
        eucl_dists,
        single_ring_dists,
        gridsize=40,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    cb1 = fig.colorbar(h1, ax=ax1)
    cb1.set_label(r"Pair Density ($\log_{10} N$)")

    ax1.set_xlabel(r"10D Euclidean Distance ($\|\mathbf{x}_i - \mathbf{x}_j\|_2$)")
    ax1.set_ylabel(r"Normalized 1D Key Distance ($\Delta k / 2^{64}$)")
    ax1.set_title("(a) Single-Curve Hilbert Mapping ($C=1$)", fontweight="bold")
    ax1.set_xlim(0, 1.8)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.25)

    # ─────────────────────────────────────────────────────────────
    # Panel 2: 3-Curve Multi-Probe Locality Preservation Heatmap
    # ─────────────────────────────────────────────────────────────
    h2 = ax2.hexbin(
        eucl_dists,
        multi_ring_dists,
        gridsize=40,
        cmap="viridis",
        mincnt=1,
        bins="log",
    )
    cb2 = fig.colorbar(h2, ax=ax2)
    cb2.set_label(r"Pair Density ($\log_{10} N$)")

    ax2.set_xlabel(r"10D Euclidean Distance ($\|\mathbf{x}_i - \mathbf{x}_j\|_2$)")
    ax2.set_ylabel(r"Multi-Probe Minimum Ring Distance ($\min_c \Delta k_c$)")
    ax2.set_title("(b) Multi-Probe Rotated Curves ($C=3$)", fontweight="bold")
    ax2.set_xlim(0, 1.8)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.25)

    # Callout highlighting cluster preservation near origin
    ax2.annotate(
        "Spatial Neighbors Clustered\nNear Ring Origin ($\\min \\Delta k \\to 0$)",
        xy=(0.25, 0.08),
        xytext=(0.45, 0.45),
        arrowprops=dict(arrowstyle="->", color="#f8fafc", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig5_hilbert_spatial_heatmap.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

    # Export LaTeX table comparing single-curve vs multi-probe locality metrics
    tex_path = out_dir / "table_fig5_hilbert_spatial_clustering.tex"
    near_mask = eucl_dists < 0.40
    med_single_near = np.median(single_ring_dists[near_mask])
    med_multi_near = np.median(multi_ring_dists[near_mask])
    p90_single_near = np.percentile(single_ring_dists[near_mask], 90)
    p90_multi_near = np.percentile(multi_ring_dists[near_mask], 90)

    # Overall correlation
    corr_single = np.corrcoef(eucl_dists, single_ring_dists)[0, 1]
    corr_multi = np.corrcoef(eucl_dists, multi_ring_dists)[0, 1]

    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{10D-to-1D Hilbert Spatial Locality Preservation Metrics}
\label{tab:hilbert_spatial_preservation}
\begin{tabular}{lcc}
\hline
\textbf{Locality Preservation Metric} & \textbf{Single Curve ($C=1$)} & \textbf{Multi-Probe Rotated ($C=3$)} \\
\hline
Overall Distance Correlation ($r$) & """ + f"{corr_single:.3f}" + r""" & """ + f"{corr_multi:.3f}" + r""" \\
Median Ring Gap for Near Neighbors ($\|\Delta \mathbf{x}\| < 0.4$) & """ + f"{med_single_near:.4f}" + r""" & """ + f"{med_multi_near:.4f}" + r""" \\
90th Percentile Ring Distortion ($P_{90}$) & """ + f"{p90_single_near:.4f}" + r""" & """ + f"{p90_multi_near:.4f}" + r""" \\
Cluster Locality Improvement Factor & Baseline (1.0$\times$) & """ + f"{med_single_near / max(1e-5, med_multi_near):.2f}" + r"""$\times$ tighter \\
\hline
\end{tabular}
\end{table}
""")
    print(f"Generated: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 10D Hilbert Spatial Locality Heatmap")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    parser.add_argument("--samples", type=int, default=250)
    args = parser.parse_args()

    generate_spatial_locality_heatmap(Path(args.out_dir), args.samples)


if __name__ == "__main__":
    main()
