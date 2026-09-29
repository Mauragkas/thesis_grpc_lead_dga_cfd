#!/usr/bin/env python3
"""
Hilbert Space-Filling Curve Encoding & Multi-Probe Key Generation Microbenchmark Plotter.
Visualizes real Criterion benchmark results for:
  - Multi-probe rotated key generation across dimensions d in {5, 10, 15}
  - Raw 10D Hilbert coordinate encoding to hex string
  - Composite DHT key parsing and gene deserialization

Generates:
  - evaluation/figures/fig6_hilbert_encoding_benchmark.png
  - evaluation/figures/table5_hilbert_spatial_indexing.tex
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


def plot_hilbert(bench_path: Path, out_dir: Path):
    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["hilbert_encoding"]

    dims = [5, 10, 15]
    keys_for_us = [h["keys_for_5d_us"], h["keys_for_10d_us"], h["keys_for_15d_us"]]
    enc_10d_ns = h["encode_hex_10d_ns"]
    parse_10d_ns = h["parse_key_10d_ns"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

    # 1. Left Panel: Multi-probe 3-curve generation vs Dimension
    ax1.plot(dims, keys_for_us, marker="o", color="#059669", linewidth=2.2, label="Multi-Probe 3-Curve Keys (`keys_for`)")
    for d, val in zip(dims, keys_for_us):
        ax1.text(d, val + 0.12, f"{val:.2f} µs", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#059669")

    ax1.set_xlabel("Gene Space Dimension ($d$)")
    ax1.set_ylabel("Key Generation Latency (µs)")
    ax1.set_title("(a) Multi-Probe Rotated Curve Scaling", fontweight="bold")
    ax1.set_xticks(dims)
    ax1.set_ylim(0, max(keys_for_us) * 1.3)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    # 2. Right Panel: Raw Encoding vs Key Deserialization Parsing
    ops = ["10D Hilbert Encode\n(`encode_hex_10d`)", "10D Key Deserialization\n(`parse_key_10d`)"]
    lat_ns = [enc_10d_ns, parse_10d_ns]
    colors = ["#3b82f6", "#f59e0b"]

    bars = ax2.bar(ops, lat_ns, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.2, width=0.45)
    for bar, val in zip(bars, lat_ns):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + 15.0, f"{val:.1f} ns",
                 ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax2.set_ylabel("Operation Latency (ns)")
    ax2.set_title("(b) Coordinate Encoding & Parsing Primitives", fontweight="bold")
    ax2.set_ylim(0, max(lat_ns) * 1.35)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "fig6_hilbert_encoding_benchmark.png"
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_png}")

    # Generate LaTeX table
    tex_path = out_dir / "table5_hilbert_spatial_indexing.tex"
    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write(r"""\begin{table}[t]
\centering
\caption{Multi-Probe Hilbert Space-Filling Curve Encoding Microbenchmarks}
\label{tab:hilbert_encoding_bench}
\begin{tabular}{llr}
\hline
\textbf{Primitive Operation} & \textbf{Configuration} & \textbf{Measured Mean Latency} \\
\hline
Raw Hilbert Coordinate Transformation & 10D Unit Hypercube $\to$ Hex String & """ + f"{enc_10d_ns:.1f}" + r"""~$\text{ns}$ \\
Composite Key Parsing \& Deserialization & 10D Key Split \& Gene Vector Decode & """ + f"{parse_10d_ns:.1f}" + r"""~$\text{ns}$ \\
\hline
Multi-Probe Rotated Curve Key Generation & 3 Curves, $d=5$ Dimensional Space & """ + f"{keys_for_us[0]:.2f}" + r"""~$\mu\text{s}$ \\
Multi-Probe Rotated Curve Key Generation & 3 Curves, $d=10$ Dimensional Space & """ + f"{keys_for_us[1]:.2f}" + r"""~$\mu\text{s}$ \\
Multi-Probe Rotated Curve Key Generation & 3 Curves, $d=15$ Dimensional Space & """ + f"{keys_for_us[2]:.2f}" + r"""~$\mu\text{s}$ \\
\hline
\end{tabular}
\end{table}
""")
    print(f"Generated: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Hilbert Encoding Microbenchmarks")
    parser.add_argument("--benchmarks", type=str, default="evaluation/data/real_criterion_benchmarks.json")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_hilbert(Path(args.benchmarks), Path(args.out_dir))


if __name__ == "__main__":
    main()
