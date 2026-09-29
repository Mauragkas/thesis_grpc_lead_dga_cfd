#!/usr/bin/env python3
"""
RMI Background Retraining, Auto-Tuning & Federated Averaging Microbenchmark Plotter.
Visualizes real Criterion benchmark results for:
  - Standard linear leaf model training vs key set size (500, 2000, 10,000 keys)
  - Mountain-climbing auto-configuration (RadixSpline vs. Linear over 1,000 and 5,000 keys)
  - Distributed Federated Averaging (fed_avg) across M in {3, 8, 16} peer DHT nodes
  - 2-bit online PID anchor adjustment loop latency

Generates:
  - evaluation/figures/fig6_rmi_train_benchmark.png
  - evaluation/figures/table4_rmi_training_and_federated_sync.tex
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


def plot_rmi_training(bench_path: Path, out_dir: Path):
    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rmi = data["rmi_train"]

    lin_sizes = np.array(rmi["train_linear_sizes"])
    lin_ms = np.array(rmi["train_linear_us"]) / 1000.0  # convert to ms

    auto_sizes = np.array(rmi["train_auto_sizes"])
    auto_ms = np.array(rmi["train_auto_us"]) / 1000.0

    nodes = np.array(rmi["fed_avg_nodes"])
    fed_us = np.array(rmi["fed_avg_ns"]) / 1000.0  # convert to us

    pid_ns = rmi["pid_adjust_single_leaf_ns"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

    # 1. Left Panel: RMI Model Training & Auto-Tuning Latency
    ax1.plot(lin_sizes, lin_ms, marker="o", color="#2563eb", linewidth=2.2, label="Linear Leaf Retraining (`RmiModel::train`)")
    ax1.plot(auto_sizes, auto_ms, marker="^", color="#d97706", linewidth=2.2, linestyle="--",
             label="Auto-Tuning Mountain Climb (`RmiModel::train_auto`)")

    ax1.set_xlabel("Dataset Size (Keys)")
    ax1.set_ylabel("Training Time (ms)")
    ax1.set_title("(a) Asynchronous RMI Retraining Cost", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    for x, y in zip(lin_sizes, lin_ms):
        ax1.text(x, y + 0.08, f"{y:.2f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2563eb")
    for x, y in zip(auto_sizes, auto_ms):
        ax1.text(x, y + 0.10, f"{y:.2f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#d97706")

    ax1.set_ylim(0, max(auto_ms) * 1.25)

    # 2. Right Panel: Federated Averaging Scaling across Peer Ring Nodes
    bars = ax2.bar([str(n) for n in nodes], fed_us, color="#8b5cf6", alpha=0.85, edgecolor="#1e293b", linewidth=1.2, width=0.45)
    ax2.set_xlabel("DHT Peer Ring Size ($M$ Nodes)")
    ax2.set_ylabel("FedAvg Synchronization Latency (µs)")
    ax2.set_title("(b) Federated Model Synchronization (`fed_avg`)", fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, val in zip(bars, fed_us):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + 0.03, f"{val:.2f} µs\n({val*1000:.0f} ns)",
                 ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    # Inset / callout box for 2-bit online PID anchor adjustment
    ax2.text(0.05, 0.85, f"2-bit PID Online Update:\n$\\mathbf{{{pid_ns:.2f}\\text{{ ns/leaf}}}}$ (~181 M ops/s)",
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecfdf5", edgecolor="#10b981", alpha=0.9),
             fontsize=9.5, fontweight="bold", color="#065f46")

    ax2.set_ylim(0, max(fed_us) * 1.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "fig6_rmi_train_benchmark.png"
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_png}")

    # Generate LaTeX table
    tex_path = out_dir / "table4_rmi_training_and_federated_sync.tex"
    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{LEAD DHT Learned Index Retraining and Federated Synchronization Benchmarks}
\label{tab:rmi_training_sync}
\begin{tabular}{llr}
\hline
\textbf{Subsystem} & \textbf{Algorithm / Routine} & \textbf{Mean Execution Time} \\
\hline
\textbf{RMI Retraining} & Linear Leaf Fitting ($K=500$ keys) & """ + f"{lin_ms[0]*1000.0:.1f}" + r"""~$\mu\text{s}$ \\
& Linear Leaf Fitting ($K=2{,}000$ keys) & """ + f"{lin_ms[1]*1000.0:.1f}" + r"""~$\mu\text{s}$ \\
& Linear Leaf Fitting ($K=10{,}000$ keys) & """ + f"{lin_ms[2]:.2f}" + r"""~$\text{ms}$ \\
& Mountain-Climbing Auto-Tune ($K=1{,}000$ keys) & """ + f"{auto_ms[0]*1000.0:.1f}" + r"""~$\mu\text{s}$ \\
& Mountain-Climbing Auto-Tune ($K=5{,}000$ keys) & """ + f"{auto_ms[1]:.2f}" + r"""~$\text{ms}$ \\
\hline
\textbf{Federated Sync} & Federated Averaging ($M=3$ DHT peers) & """ + f"{fed_us[0]*1000.0:.1f}" + r"""~$\text{ns}$ \\
& Federated Averaging ($M=8$ DHT peers) & """ + f"{fed_us[1]*1000.0:.1f}" + r"""~$\text{ns}$ \\
& Federated Averaging ($M=16$ DHT peers) & """ + f"{fed_us[2]:.2f}" + r"""~$\mu\text{s}$ \\
& 2-bit Online PID Anchor Adjustment & """ + f"{pid_ns:.2f}" + r"""~$\text{ns}$ \\
\hline
\end{tabular}
\end{table}
""")
    print(f"Generated: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot RMI Training & FedAvg Microbenchmarks")
    parser.add_argument("--benchmarks", type=str, default="evaluation/data/real_criterion_benchmarks.json")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_rmi_training(Path(args.benchmarks), Path(args.out_dir))


if __name__ == "__main__":
    main()
