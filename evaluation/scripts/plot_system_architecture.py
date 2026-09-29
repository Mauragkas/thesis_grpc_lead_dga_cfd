#!/usr/bin/env python3
"""
System Architecture & Topology Diagram Generator.
Renders the complete end-to-end distributed system architecture for the thesis:
  1. Distributed Chord Ring (4 Island Orchestrators + Base-10 finger routing + LeadMigration)
  2. Envoy Proxy (gRPC Least-Request Multiplexer)
  3. Distributed Worker Pool (AeroSandbox Vortex Lattice CFD evaluators)
  4. Surrogate Cache / Inference Engine (Tier-1 exact hash + Tier-2 MLP)
  5. Telemetry & Observation Bus (Fluent-Bit / Kafka event streaming)

Generates:
  - evaluation/figures/fig5_system_architecture.png
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# IEEE publication aesthetics
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.autolayout": True,
})


def draw_system_architecture(out_dir: Path):
    fig, ax = plt.subplots(figsize=(13.0, 7.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Helper styling function
    def draw_box(x, y, w, h, title, subtitle="", color="#f8fafc", edge="#64748b", lw=1.5, rx=1.5):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.2,rounding_size={rx}",
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w / 2, y + h - 2.8, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a", zorder=3)
            ax.text(x + w / 2, y + (h - 2.8) / 2, subtitle, ha="center", va="center", fontsize=8.0, color="#475569", zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=9.0, fontweight="bold", color="#0f172a", zorder=3)
        return box

    def draw_arrow(x1, y1, x2, y2, label="", color="#2563eb", lw=1.6, ls="-", rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=connectionstyle,
            arrowstyle="->,head_width=3.5,head_length=5",
            color=color,
            linewidth=lw,
            linestyle=ls,
            zorder=4,
        )
        ax.add_patch(arrow)
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + 1.6
            ax.text(mx, my, label, ha="center", va="center", fontsize=7.5, fontweight="bold", color=color, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="none", alpha=0.85))

    # ─────────────────────────────────────────────────────────────────
    # Zone 1: Distributed Chord Island Ring (Left)
    # ─────────────────────────────────────────────────────────────────
    zone_ring = patches.FancyBboxPatch((2, 18), 38, 76, boxstyle="round,pad=0.8,rounding_size=2.0",
                                      facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(zone_ring)
    ax.text(21, 92, "LEAD Distributed Chord Ring (Orchestrators)", ha="center", fontsize=11, fontweight="bold", color="#1e40af")

    # 4 Island Nodes in a Ring
    draw_box(13, 76, 16, 10, "Island 0 (Lead)", "GeneStore + RingMember\nBase-10 Fingers", color="#dbeafe", edge="#3b82f6")
    draw_box(24, 52, 14, 9, "Island 1", "GaRunner Pop=60\nTop-K MigrantHook", color="#ffffff", edge="#60a5fa")
    draw_box(13, 26, 16, 9, "Island 2", "Hilbert Key Hash\nRMI Learned Index", color="#ffffff", edge="#60a5fa")
    draw_box(4, 52, 14, 9, "Island 3", "Tokio Async Engine\nLocal Population", color="#ffffff", edge="#60a5fa")

    # Ring Migration Links (Clockwise)
    draw_arrow(21, 76, 28, 61, "Migration\n(M_int=5)", color="#2563eb", rad=0.1)
    draw_arrow(28, 52, 21, 35, "", color="#2563eb", rad=0.1)
    draw_arrow(13, 30, 9, 52, "Successor\nRing", color="#2563eb", rad=0.1)
    draw_arrow(11, 61, 16, 76, "", color="#2563eb", rad=0.1)

    # ─────────────────────────────────────────────────────────────────
    # Zone 2: Intelligent Tier Evaluation & Load Balancing (Center)
    # ─────────────────────────────────────────────────────────────────
    zone_tier = patches.FancyBboxPatch((43, 40), 22, 54, boxstyle="round,pad=0.8,rounding_size=2.0",
                                       facecolor="#fefce8", edgecolor="#fef08a", linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(zone_tier)
    ax.text(54, 92, "Evaluation Pipeline & Envoy", ha="center", fontsize=11, fontweight="bold", color="#854d0e")

    draw_box(46, 76, 16, 10, "Tier 1 Cache", "Exact Gene Match\n10D Hilbert Spatial Hash", color="#ffffff", edge="#eab308")
    draw_box(46, 61, 16, 9, "Tier 2 Surrogate", "PyTorch CUDA MLP\nPredictive Inference", color="#ffffff", edge="#eab308")
    draw_box(46, 45, 16, 10, "Envoy Proxy", "Least-Request LB\ngRPC Multiplexing (50051)", color="#fef08a", edge="#ca8a04", lw=2)

    # Cache hit / fall-through arrows
    draw_arrow(29, 81, 46, 81, "Evaluate Gen", color="#0284c7")
    draw_arrow(54, 76, 54, 70, "Cache Miss", color="#d97706")
    draw_arrow(54, 61, 54, 55, "Uncertainty Fallback", color="#d97706")

    # ─────────────────────────────────────────────────────────────────
    # Zone 3: Distributed AeroSandbox Worker Pool (Right)
    # ─────────────────────────────────────────────────────────────────
    zone_workers = patches.FancyBboxPatch((68, 35), 30, 59, boxstyle="round,pad=0.8,rounding_size=2.0",
                                         facecolor="#ecfdf5", edgecolor="#a7f3d0", linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(zone_workers)
    ax.text(83, 92, "Worker Pool (Tier 3 CFD)", ha="center", fontsize=11, fontweight="bold", color="#065f46")

    draw_box(71, 78, 24, 8, "CFD Worker 1", "AeroSandbox VLM / Trim Polar", color="#ffffff", edge="#10b981")
    draw_box(71, 66, 24, 8, "CFD Worker 2", "AeroSandbox VLM / Trim Polar", color="#ffffff", edge="#10b981")
    draw_box(71, 54, 24, 8, "CFD Worker 3", "AeroSandbox VLM / Trim Polar", color="#ffffff", edge="#10b981")
    draw_box(71, 42, 24, 8, "CFD Worker N (W=32)", "Scale-out Autonomous Workers", color="#d1fae5", edge="#059669")

    # Envoy dispatch arrows to workers
    draw_arrow(62, 50, 71, 82, "gRPC Batch", color="#059669")
    draw_arrow(62, 50, 71, 70, "", color="#059669")
    draw_arrow(62, 50, 71, 58, "", color="#059669")
    draw_arrow(62, 50, 71, 46, "Least-Request", color="#059669")

    # ─────────────────────────────────────────────────────────────────
    # Zone 4: Telemetry & Monitoring Pipeline (Bottom)
    # ─────────────────────────────────────────────────────────────────
    zone_telem = patches.FancyBboxPatch((2, 2), 96, 12, boxstyle="round,pad=0.8,rounding_size=2.0",
                                        facecolor="#fdf4ff", edgecolor="#f0abfc", linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(zone_telem)
    ax.text(50, 11.5, "Telemetry, Observation & Streaming Pipeline", ha="center", fontsize=11, fontweight="bold", color="#86198f")

    draw_box(6, 4, 18, 6, "Protobuf Telemetry", "1.03 us Serialization", color="#ffffff", edge="#c084fc")
    draw_box(29, 4, 18, 6, "Fluent-Bit Daemon", "Log & Metric Collector", color="#ffffff", edge="#c084fc")
    draw_box(52, 4, 18, 6, "Apache Kafka", "Generation Events Bus", color="#fae8ff", edge="#a855f7")
    draw_box(75, 4, 20, 6, "Grafana / Ingestion", "Real-Time Telemetry Dashboard", color="#ffffff", edge="#c084fc")

    draw_arrow(24, 7, 29, 7, "", color="#9333ea")
    draw_arrow(47, 7, 52, 7, "Stream", color="#9333ea")
    draw_arrow(70, 7, 75, 7, "Consumer", color="#9333ea")

    # Feed into telemetry bus from orchestrator
    draw_arrow(18, 26, 15, 10, "GenerationRecord", color="#9333ea", rad=-0.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig5_system_architecture.png"
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate System Architecture Diagram")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    draw_system_architecture(Path(args.out_dir))


if __name__ == "__main__":
    main()
