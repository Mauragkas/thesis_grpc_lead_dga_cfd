#!/usr/bin/env python3
"""
Envoy gRPC Load Balancing & Worker Latency Distribution Plotter.
Plots real empirical AeroSandbox CFD worker latency distributions:
  - Round-Robin scheduling vs. Envoy Least-Request (P2C)
  - Latency percentiles (P50, P90, P99)
  - Per-worker request fairness across N=4 worker backends

Generates:
  - evaluation/figures/fig3_envoy_load_balancing.png
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


def jains_fairness_index(x: np.ndarray) -> float:
    return float((np.sum(x) ** 2) / (len(x) * np.sum(x ** 2)))


def plot_real_load_balancing(data_path: Path, out_dir: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rr_lat = np.array(data["round_robin_latencies"])
    lr_lat = np.array(data["least_request_latencies"])
    rr_counts = np.array(data["round_robin_counts"])
    lr_counts = np.array(data["least_request_counts"])

    p50_rr, p90_rr, p99_rr = np.percentile(rr_lat, [50, 90, 99])
    p50_lr, p90_lr, p99_lr = np.percentile(lr_lat, [50, 90, 99])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    # ─────────────────────────────────────────────────────────────────
    # Panel 1: Empirical CFD Worker Latency CDF & Tail Straggler Reduction
    # ─────────────────────────────────────────────────────────────────
    sorted_rr = np.sort(rr_lat)
    sorted_lr = np.sort(lr_lat)
    cdf_rr = np.arange(1, len(sorted_rr) + 1) / len(sorted_rr)
    cdf_lr = np.arange(1, len(sorted_lr) + 1) / len(sorted_lr)

    ax1.plot(sorted_lr, cdf_lr, color="#10b981", linewidth=2.2, label=f"Envoy Least-Request ($P_{{50}}={p50_lr:.0f}$, $P_{{99}}={p99_lr:.0f}$ ms)")
    ax1.plot(sorted_rr, cdf_rr, color="#ef4444", linewidth=1.8, linestyle="--", label=f"Static Round-Robin ($P_{{50}}={p50_rr:.0f}$, $P_{{99}}={p99_rr:.0f}$ ms)")

    ax1.axvline(x=p99_lr, color="#10b981", linestyle=":", alpha=0.7)
    ax1.axvline(x=p99_rr, color="#ef4444", linestyle=":", alpha=0.7)
    tail_cut = p99_rr - p99_lr

    ax1.annotate(
        f"$P_{{99}}$ Tail Cut:\n$-{tail_cut:.0f}$ ms",
        xy=(p99_lr, 0.99),
        xytext=(p99_lr - 80, 0.70),
        arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    ax1.set_xlabel("CFD Evaluation Latency (ms)")
    ax1.set_ylabel("Cumulative Probability ($P(X \\leq t)$)")
    ax1.set_title("(a) Empirical Worker Latency Profile ($P_{50}, P_{90}, P_{99}$)", fontweight="bold")
    ax1.set_xlim(150, 450)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.5)

    # ─────────────────────────────────────────────────────────────────
    # Panel 2: Per-Worker Evaluation Counts & Load Fairness
    # ─────────────────────────────────────────────────────────────────
    workers = ["Worker 1\n(:50051)", "Worker 2\n(:50052)", "Worker 3\n(:50053)", "Worker 4\n(:50054)"]
    x = np.arange(len(workers))
    width = 0.35

    jain_rr = jains_fairness_index(rr_counts)
    jain_lr = jains_fairness_index(lr_counts)

    b1 = ax2.bar(x - width / 2, rr_counts, width, label=f"Static Round-Robin ($J={jain_rr:.4f}$)", color="#94a3b8", edgecolor="black", linewidth=0.6, alpha=0.9)
    b2 = ax2.bar(x + width / 2, lr_counts, width, label=f"Envoy Least-Request ($J={jain_lr:.4f}$)", color="#38bdf8", edgecolor="black", linewidth=0.6, alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(workers, fontsize=10.5)
    ax2.set_ylabel("Assigned Evaluations (Total $N = 100$)")
    ax2.set_title("(b) Backend Worker Workload Fairness", fontweight="bold")
    ax2.set_ylim(0, max(np.max(rr_counts), np.max(lr_counts)) * 1.35)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.8, f"{int(h)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig3_envoy_load_balancing.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Real Envoy Load Balancing Benchmark")
    parser.add_argument(
        "--data",
        type=str,
        default="evaluation/data/real_worker_balancing_benchmarks.json",
        help="Path to real_worker_balancing_benchmarks.json",
    )
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_real_load_balancing(Path(args.data), Path(args.out_dir))


if __name__ == "__main__":
    main()
