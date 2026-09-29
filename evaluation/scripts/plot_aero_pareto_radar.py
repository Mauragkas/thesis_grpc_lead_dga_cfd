#!/usr/bin/env python3
"""
Aerodynamic Multi-Objective Trade-offs: Pareto Front & Radar Benchmark Plotter.
Uses real `AerosandboxAeroEvaluator` and `FitnessEvaluator` to evaluate candidate designs across:
  1. Aerodynamic Efficiency (Lift-to-Drag ratio L/D)
  2. Structural Mass / Weight (kg)
  3. Fuselage Internal Volume (cm^3)
  4. Pitch Stability Metric (-C_m_alpha)
  5. Wing Aspect Ratio (AR)

Generates:
  - evaluation/figures/fig5_aero_pareto_radar.png
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "src"))
from worker.config import load_config
from worker.aero import AerosandboxAeroEvaluator
from worker.fitness import FitnessEvaluator

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


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Find the pareto-efficient points (assuming minimizing costs)."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def generate_pareto_radar_chart(out_dir: Path, n_candidates: int = 50):
    print(f"Evaluating {n_candidates} wings using AeroSandbox for Multi-Objective Trade-offs...")
    cfg = load_config()
    aero = AerosandboxAeroEvaluator(cfg)
    fe = FitnessEvaluator(aero, cfg)

    base_genes = np.array([120.0, 45.0, 22.0, 12.5, 3.0, -2.0, 25.0, 240.0, 25.0, 2.5, 12.0])
    rng = np.random.default_rng(123)

    records = []

    # Baseline wing
    b_out = fe.evaluate_genes(base_genes)
    if b_out.aero:
        records.append({
            "name": "Baseline",
            "ld": b_out.aero.ld,
            "mass": b_out.aero.mass_kg,
            "volume_cm3": b_out.fuselage_volume_mm3 / 1000.0,
            "cm_alpha": -b_out.aero.cm_alpha,
            "ar": (2.0 * 240.0) / ((120.0 + 45.0) / 2.0),
        })

    # Sample variants across span, chord, and sweep
    for i in range(n_candidates):
        jitter = rng.normal(0.0, 0.12, size=len(base_genes))
        genes = (base_genes * (1.0 + jitter)).tolist()
        out = fe.evaluate_genes(genes)
        if out.aero and not out.rejected and out.aero.ld > 4.0:
            span = genes[7] * 2.0
            root_c = genes[0]
            tip_c = genes[1]
            c_mean = (root_c + tip_c) / 2.0
            ar = span / c_mean

            records.append({
                "name": f"Variant_{i}",
                "ld": out.aero.ld,
                "mass": out.aero.mass_kg,
                "volume_cm3": out.fuselage_volume_mm3 / 1000.0,
                "cm_alpha": -out.aero.cm_alpha,
                "ar": ar,
            })

    print(f"Successfully evaluated {len(records)} feasible candidate wings.")

    # ─────────────────────────────────────────────────────────────────
    # Pareto Front Extraction: Maximize L/D, Minimize Mass
    # ─────────────────────────────────────────────────────────────────
    all_ld = np.array([r["ld"] for r in records])
    all_mass = np.array([r["mass"] for r in records])
    all_vol = np.array([r["volume_cm3"] for r in records])

    # Costs matrix for Pareto: [-L/D, +Mass]
    costs = np.column_stack([-all_ld, all_mass])
    pareto_mask = is_pareto_efficient(costs)

    fig = plt.figure(figsize=(13.5, 5.0), dpi=300)

    # Panel 1: 2D Pareto Scatter Plot (L/D vs. Mass vs. Fuselage Volume)
    ax1 = fig.add_subplot(1, 2, 1)

    sc = ax1.scatter(
        all_mass * 1000.0,
        all_ld,
        c=all_vol,
        cmap="coolwarm",
        s=45,
        alpha=0.7,
        edgecolors="none",
        label="Dominated Designs",
    )
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label("Fuselage Volume ($V_{\\mathrm{fuse}}$, cm$^3$)")

    # Plot Pareto Front
    p_mass = all_mass[pareto_mask] * 1000.0
    p_ld = all_ld[pareto_mask]
    sort_idx = np.argsort(p_mass)
    ax1.plot(
        p_mass[sort_idx],
        p_ld[sort_idx],
        "r--o",
        linewidth=2.0,
        markersize=6,
        label="Pareto Efficient Frontier",
    )

    # Highlight Optimal Compromise & Baseline
    ax1.scatter([all_mass[0] * 1000.0], [all_ld[0]], color="#10b981", s=110, marker="*", edgecolor="black", label="Baseline Design", zorder=5)

    ax1.set_xlabel("Wing Structural Mass (grams)")
    ax1.set_ylabel("Lift-to-Drag Ratio ($L/D$)")
    ax1.set_title("(a) Aerodynamic Pareto Trade-off Front", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.0)

    # ─────────────────────────────────────────────────────────────────
    # Panel 2: Multi-Objective Radar Chart
    # ─────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, polar=True)

    categories = [
        "Aerodynamic\nEfficiency ($L/D$)",
        "Lightweight\n($1 / \\mathrm{Mass}$)",
        "Fuselage\nCapacity ($V$)",
        "Pitch\nStability ($-C_{m_\\alpha}$)",
        "Aspect\nRatio ($\\mathrm{AR}$)",
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Select 3 designs: Baseline, Best L/D (High Aspect Ratio), and Pareto Compromise
    best_ld_idx = np.argmax(all_ld)
    pareto_indices = np.where(pareto_mask)[0]
    compromise_idx = pareto_indices[len(pareto_indices) // 2]

    # Normalize metrics [0.1, 1.0] for radar comparison
    def get_normalized_vector(idx):
        r = records[idx]
        v_ld = r["ld"] / np.max(all_ld)
        v_light = (1.0 / r["mass"]) / np.max(1.0 / all_mass)
        v_vol = r["volume_cm3"] / np.max(all_vol)
        v_stab = max(0.1, r["cm_alpha"] / max(0.01, np.max([rec["cm_alpha"] for rec in records])))
        v_ar = r["ar"] / np.max([rec["ar"] for rec in records])
        vals = [v_ld, v_light, v_vol, v_stab, v_ar]
        vals += vals[:1]
        return vals

    v_base = get_normalized_vector(0)
    v_best_ld = get_normalized_vector(best_ld_idx)
    v_comp = get_normalized_vector(compromise_idx)

    ax2.plot(angles, v_base, "o-", color="#10b981", linewidth=1.8, label="Baseline")
    ax2.fill(angles, v_base, color="#10b981", alpha=0.15)

    ax2.plot(angles, v_best_ld, "s-", color="#2563eb", linewidth=1.8, label="Max $L/D$ Specialization")
    ax2.fill(angles, v_best_ld, color="#2563eb", alpha=0.15)

    ax2.plot(angles, v_comp, "^-", color="#dc2626", linewidth=2.0, label="Pareto Balanced Elite")
    ax2.fill(angles, v_comp, color="#dc2626", alpha=0.20)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9.5)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("(b) Multi-Objective Radar Comparison", fontweight="bold", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig5_aero_pareto_radar.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Aero Multi-Objective Pareto & Radar Charts")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    parser.add_argument("--candidates", type=int, default=50)
    args = parser.parse_args()

    generate_pareto_radar_chart(Path(args.out_dir), args.candidates)


if __name__ == "__main__":
    main()
