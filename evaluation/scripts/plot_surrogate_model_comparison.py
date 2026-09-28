#!/usr/bin/env python3
"""
Publication-ready 6-Panel Surrogate Architecture & Model Comparison.
Evaluates and compares:
  1. Gaussian Process (Matérn 5/2)
  2. Deep Neural Network (MLP)
  3. Random Forest (RF)
  4. k-Nearest Neighbors (k-NN)

across Accuracy (R², RMSE, MAE), Computational Latency, Inference Throughput,
Parity Overlays, Residual Error Distributions, and Empirical Learning Curves.

Generates:
  - evaluation/figures/fig2_surrogate_model_comparison.png
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
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
})

COLORS = {
    "Gaussian Process": "#2563eb",     # Blue
    "k-Nearest": "#10b981",            # Emerald
    "Random Forest": "#f59e0b",        # Amber
    "Neural Network": "#8b5cf6",       # Purple
    "MLP": "#8b5cf6",
}


def model_color(name: str) -> str:
    for key, color in COLORS.items():
        if key.lower() in name.lower():
            return color
    return "#6b7280"


def load_report(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def short_name(full_name: str) -> str:
    if "gaussian" in full_name.lower():
        return "GP (Matérn 5/2)"
    if "k-nearest" in full_name.lower() or "knn" in full_name.lower():
        return "k-NN"
    if "random forest" in full_name.lower() or "rf" in full_name.lower():
        return "Random Forest"
    if "neural network" in full_name.lower() or "mlp" in full_name.lower():
        return "Neural Net (MLP)"
    return full_name.split("(")[0].strip()


def plot_model_comparison(data: dict, out_dir: Path):
    models = data["models"]

    fig = plt.figure(figsize=(20, 11.5), constrained_layout=True, dpi=300)
    gs = fig.add_gridspec(2, 3)

    names = [m["name"] for m in models]
    snames = [short_name(n) for n in names]
    colors = [model_color(n) for n in names]
    x_pos = np.arange(len(names))

    # ── Panel 1: Accuracy (R² and RMSE grouped bar) ───────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    w = 0.35
    r2s = [m["test_r2"] for m in models]
    rmses = [m["test_rmse"] for m in models]
    b1 = ax1.bar(x_pos - w / 2, r2s, w, label=r"Test $R^2$", color=colors, alpha=0.9, edgecolor="black", linewidth=0.7)
    b2 = ax1.bar(x_pos + w / 2, rmses, w, label="Test RMSE", color=colors, alpha=0.55, edgecolor="black", linewidth=0.7, hatch="//")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(snames, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Metric Score", fontsize=11)
    ax1.set_title("(a) Accuracy on Holdout Test Set ($N=241$)", fontsize=12, fontweight="bold")
    max_acc_val = max(max(r2s), max(rmses))
    ax1.set_ylim(0.0, max_acc_val * 1.25)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # ── Panel 2: Computational latency (log scale) ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    train_ms = [m["train_ms"] for m in models]
    infer_ms = [m["infer_ms_per_1k"] for m in models]
    b3 = ax2.bar(x_pos - w / 2, train_ms, w, label="Training (ms)", color=colors, alpha=0.9, edgecolor="black", linewidth=0.7)
    b4 = ax2.bar(x_pos + w / 2, infer_ms, w, label="Inference 1k pts (ms)", color=colors, alpha=0.55, edgecolor="black", linewidth=0.7, hatch="//")
    ax2.set_yscale("log")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(snames, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Wall-Clock Time (ms, log scale)", fontsize=11)
    ax2.set_title("(b) Computational Latency Comparison", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6, which="both")
    for rect in list(b3) + list(b4):
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h * 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # ── Panel 3: Throughput ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    tput = [m["throughput_evals_per_sec"] / 1000.0 for m in models]  # k evals/sec
    bars = ax3.bar(snames, tput, color=colors, edgecolor="black", linewidth=0.7, alpha=0.9)
    ax3.set_ylabel("Inference Throughput (k designs / sec)", fontsize=11)
    ax3.set_title("(c) Surrogate Inference Throughput", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    for bar, val in zip(bars, tput):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + max(tput) * 0.01, f"{val:.1f}k", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Panel 4: Parity overlay ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    all_y = np.concatenate([m["y_test"] for m in models])
    lo, hi = np.min(all_y) - 0.3, np.max(all_y) + 0.3
    ax4.plot([lo, hi], [lo, hi], "k--", lw=1.6, label="1:1 ideal parity")
    for m in models:
        ax4.scatter(m["y_test"], m["y_pred"], s=22, alpha=0.6, color=model_color(m["name"]), label=f"{short_name(m['name'])} ($R^2={m['test_r2']:.3f}$)")
    ax4.set_xlabel("VLM Ground Truth Fitness", fontsize=11)
    ax4.set_ylabel("Surrogate Predicted Fitness", fontsize=11)
    ax4.set_title("(d) Parity Plot — Holdout Test Set", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8.5)
    ax4.grid(linestyle=":", alpha=0.6)

    # ── Panel 5: Residual histograms ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-4.0, 4.0, 28)
    for m in models:
        residuals = np.array(m["y_test"]) - np.array(m["y_pred"])
        ax5.hist(residuals, bins=bins, alpha=0.45, color=model_color(m["name"]), edgecolor=model_color(m["name"]), label=f"{short_name(m['name'])} (MAE={m['test_mae']:.3f})")
    ax5.axvline(0.0, color="red", linestyle="--", lw=1.3)
    ax5.set_xlabel("Residual (True − Predicted)", fontsize=11)
    ax5.set_ylabel("Count", fontsize=11)
    ax5.set_title("(e) Residual Error Distributions", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=8.5)
    ax5.grid(linestyle=":", alpha=0.6)

    # ── Panel 6: Learning curves ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    for m in models:
        curve = m.get("learning_curve", [])
        if not curve:
            continue
        sizes = [pt[0] for pt in curve]
        r2vals = [pt[1] for pt in curve]
        ax6.plot(sizes, r2vals, "o-", lw=2.0, color=model_color(m["name"]), label=short_name(m["name"]))
    ax6.set_xlabel("Training Set Size (N)", fontsize=11)
    ax6.set_ylabel(r"Holdout Test $R^2$ Score", fontsize=11)
    ax6.set_title("(f) Sample Efficiency / Learning Curves", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(linestyle=":", alpha=0.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig2_surrogate_model_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Surrogate Model Comparison")
    parser.add_argument(
        "--json-report",
        type=str,
        default="evaluation/data/surrogate_compare_results.json",
        help="Path to compare_results.json",
    )
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    json_path = Path(args.json_report)
    if not json_path.exists():
        json_path = Path("surrogate_node/scripts/compare_results.json")

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    data = load_report(json_path)
    plot_model_comparison(data, Path(args.out_dir))


if __name__ == "__main__":
    main()
