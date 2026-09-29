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
  - evaluation/figures/table_fig5_aero_pareto_comparison.tex
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "src"))
from worker.config import load_config, GENE_BOUNDS
from worker.aero import AerosandboxAeroEvaluator
from worker.fitness import FitnessEvaluator
from worker.geometry import decode_genes

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
    print(f"Evaluating {n_candidates} wings using AeroSandbox for Multi-Objective Trade-offs (10D normalized)...")
    cfg = load_config()
    aero = AerosandboxAeroEvaluator(cfg)
    fe = FitnessEvaluator(aero, cfg)

    # Optimal converged genome as base
    base_genes = np.array([0.887, 0.958, 0.166, 0.05, 0.873, 0.699, 0.083, 0.966, 0.512, 0.018])
    rng = np.random.default_rng(123)

    records = []

    # Baseline wing
    b_out = fe.evaluate_genes(base_genes)
    if b_out.aero:
        b_params = decode_genes(base_genes, bounds=GENE_BOUNDS)
        b_span = b_params["wing_span"] * 2.0
        b_c_mean = (b_params["wing_root_chord"] + b_params["wing_tip_chord"]) / 2.0
        b_ar = b_span / b_c_mean
        records.append({
            "name": "Baseline",
            "ld": b_out.aero.ld,
            "mass": b_out.aero.mass_kg,
            "volume_cm3": b_out.fuselage_volume_mm3 / 1000.0,
            "cm_alpha": -b_out.aero.cm_alpha,
            "ar": b_ar,
        })

    # Sample variants across normalized design space
    for i in range(n_candidates):
        jitter = rng.normal(0.0, 0.12, size=len(GENE_BOUNDS))
        genes = np.clip(base_genes + jitter, 0.0, 1.0).tolist()
        out = fe.evaluate_genes(genes)
        if out.aero and not out.rejected and out.aero.ld > 4.0:
            params = decode_genes(genes, bounds=GENE_BOUNDS)
            span = params["wing_span"] * 2.0
            root_c = params["wing_root_chord"]
            tip_c = params["wing_tip_chord"]
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
        cmap="viridis",
        s=45,
        alpha=0.75,
        edgecolors="none",
        label="Evaluated Candidates",
    )
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label("Fuselage Volume ($V_{\\mathrm{fuse}}$, $\\mathrm{cm}^3$)")

    # Connect Pareto Front with Step Line
    p_indices = np.where(pareto_mask)[0]
    p_sorted = p_indices[np.argsort(all_mass[p_indices])]
    ax1.plot(
        all_mass[p_sorted] * 1000.0,
        all_ld[p_sorted],
        color="#dc2626",
        linestyle="--",
        linewidth=2.0,
        label="Empirical Pareto Boundary",
        zorder=5,
    )
    ax1.scatter(
        all_mass[p_sorted] * 1000.0,
        all_ld[p_sorted],
        color="#dc2626",
        marker="D",
        s=60,
        edgecolors="black",
        linewidths=0.8,
        label="Non-Dominated Solutions",
        zorder=6,
    )

    ax1.scatter(
        records[0]["mass"] * 1000.0,
        records[0]["ld"],
        color="#10b981",
        marker="*",
        s=220,
        edgecolors="black",
        linewidths=1.2,
        label="Baseline Reference Design",
        zorder=7,
    )

    ax1.set_xlabel("Total Structural Mass ($m$, grams)")
    ax1.set_ylabel("Lift-to-Drag Ratio ($L/D$)")
    ax1.set_title("(a) Multi-Objective Aerodynamic Pareto Front", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.0)

    # ─────────────────────────────────────────────────────────────
    # Panel 2: Multi-Objective Radar Chart
    # ─────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, polar=True)

    categories = [
        "Aerodynamic\nEfficiency ($L/D$)",
        r"Lightweight\n($1 / \mathrm{Mass}$)",
        r"Fuselage\nCapacity ($V$)",
        r"Pitch\nStability ($-C_{m_\alpha}$)",
        r"Aspect\nRatio ($\mathrm{AR}$)",
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Select 3 designs: Baseline, Best L/D, and Pareto Compromise
    best_ld_idx = np.argmax(all_ld)
    pareto_indices = np.where(pareto_mask)[0]
    compromise_idx = pareto_indices[len(pareto_indices) // 2]

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

    ax2.plot(angles, v_best_ld, "s-", color="#2563eb", linewidth=1.8, label=r"Max $L/D$ Specialization")
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

    # Export LaTeX table comparing Baseline, Max L/D, and Pareto Balanced Elite
    tex_path = out_dir / "table_fig5_aero_pareto_comparison.tex"
    b_rec = records[0]
    m_rec = records[best_ld_idx]
    c_rec = records[compromise_idx]

    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{Multi-Objective Aerodynamic Performance Comparison across Pareto Candidates}
\label{tab:aero_pareto_comparison}
\begin{tabular}{lccc}
\hline
\textbf{Design Objective / Metric} & \textbf{Baseline Wing} & \textbf{Max $L/D$ Specialization} & \textbf{Pareto Balanced Elite} \\
\hline
Aerodynamic Efficiency ($L/D$) & """ + f"{b_rec['ld']:.2f}" + r""" & """ + f"{m_rec['ld']:.2f}" + r""" & """ + f"{c_rec['ld']:.2f}" + r""" \\
Structural Mass ($m$, grams) & """ + f"{b_rec['mass']*1000.0:.1f}" + r"""~g & """ + f"{m_rec['mass']*1000.0:.1f}" + r"""~g & """ + f"{c_rec['mass']*1000.0:.1f}" + r"""~g \\
Aspect Ratio ($\mathrm{AR}$) & """ + f"{b_rec['ar']:.2f}" + r""" & """ + f"{m_rec['ar']:.2f}" + r""" & """ + f"{c_rec['ar']:.2f}" + r""" \\
Fuselage Volume ($V$, $\mathrm{cm}^3$) & """ + f"{b_rec['volume_cm3']:.1f}" + r""" & """ + f"{m_rec['volume_cm3']:.1f}" + r""" & """ + f"{c_rec['volume_cm3']:.1f}" + r""" \\
Static Pitch Stability ($-C_{m_\alpha}$) & """ + f"{b_rec['cm_alpha']:.3f}" + r""" & """ + f"{m_rec['cm_alpha']:.3f}" + r""" & """ + f"{c_rec['cm_alpha']:.3f}" + r""" \\
Relative $L/D$ Gain over Baseline & Baseline & """ + f"+{((m_rec['ld'] - b_rec['ld']) / b_rec['ld']) * 100.0:.1f}\\%" + r""" & """ + f"+{((c_rec['ld'] - b_rec['ld']) / b_rec['ld']) * 100.0:.1f}\\%" + r""" \\
\hline
\end{tabular}
\end{table}
""")
    print(f"Generated: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Aero Multi-Objective Pareto & Radar Charts")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    parser.add_argument("--candidates", type=int, default=50)
    args = parser.parse_args()

    generate_pareto_radar_chart(Path(args.out_dir), args.candidates)


if __name__ == "__main__":
    main()
