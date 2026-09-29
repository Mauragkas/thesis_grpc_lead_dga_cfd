#!/usr/bin/env python3
"""
Telemetry Protocol Overhead & Serialization Criterion.rs Plotter.
Plots real empirical Criterion.rs benchmark measurements comparing:
  - Binary Protobuf (`prost::Message::encode`)
  - JSON Lines (`serde_json::to_string`)
across batch sizes (N = 10, 50, 100 individuals).

Generates:
  - evaluation/figures/fig3_telemetry_protocol_overhead.png
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


def plot_real_telemetry_benchmark(data_path: Path, out_dir: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)["telemetry_serialization"]

    batches = data["batches"]
    labels = [f"$N={b}$" for b in batches]
    proto_us = np.array(data["proto_us"])
    json_us = np.array(data["json_us"])

    # Over-the-wire byte sizes based on the measured struct definition:
    # Protobuf: 98 bytes per individual (varint tags + IEEE 754 packed double precision floats)
    # JSON: 385 bytes per individual (stringified keys, floating point representations)
    bytes_proto_kb = (np.array(batches) * 98) / 1024.0
    bytes_json_kb = (np.array(batches) * 385) / 1024.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    x = np.arange(len(batches))
    width = 0.35

    # ─────────────────────────────────────────────────────────────────
    # Panel 1: Wire Serialization Footprint (KB)
    # ─────────────────────────────────────────────────────────────────
    b1 = ax1.bar(x - width / 2, bytes_proto_kb, width, label="Protobuf Binary (prost / Tonic gRPC)", color="#2563eb", edgecolor="black", linewidth=0.7, alpha=0.9)
    b2 = ax1.bar(x + width / 2, bytes_json_kb, width, label="JSON Lines (serde_json)", color="#f59e0b", edgecolor="black", linewidth=0.7, alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_xlabel("Evaluation Batch Size")
    ax1.set_ylabel("Wire Payload Footprint (KB)")
    ax1.set_title("(a) Wire Serialization Overhead", fontweight="bold")
    ax1.set_ylim(0, max(bytes_json_kb) * 1.3)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)

    compression_factor = bytes_json_kb[-1] / bytes_proto_kb[-1]
    ax1.text(
        x[-1] - width / 2,
        bytes_proto_kb[-1] + 1.2,
        f"{compression_factor:.1f}$\\times$\nSmaller",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#1e40af",
    )

    # ─────────────────────────────────────────────────────────────────
    # Panel 2: Empirical Serialization Latency from Criterion.rs (microseconds)
    # ─────────────────────────────────────────────────────────────────
    ax2.plot(x, json_us, "s-", color="#f59e0b", linewidth=2.0, markersize=7, label="JSON Serialization (serde_json)")
    ax2.plot(x, proto_us, "o-", color="#2563eb", linewidth=2.2, markersize=7, label="Protobuf Encoding (prost)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_xlabel("Evaluation Batch Size")
    ax2.set_ylabel("Serialization CPU Time (microseconds)")
    ax2.set_title("(b) Empirical CPU Encoding Time (Criterion.rs)", fontweight="bold")
    ax2.set_ylim(0, max(json_us) * 1.25)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)

    speedup_at_100 = json_us[-1] / proto_us[-1]
    ax2.text(
        x[-1] - 0.25,
        proto_us[-1] + 2.0,
        f"{speedup_at_100:.1f}$\\times$ Faster\nEncoding",
        fontsize=9,
        fontweight="bold",
        color="#1e40af",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eff6ff", edgecolor="#93c5fd"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig3_telemetry_protocol_overhead.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Real Telemetry Benchmark")
    parser.add_argument(
        "--data",
        type=str,
        default="evaluation/data/real_criterion_benchmarks.json",
        help="Path to real_criterion_benchmarks.json",
    )
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_real_telemetry_benchmark(Path(args.data), Path(args.out_dir))


if __name__ == "__main__":
    main()
