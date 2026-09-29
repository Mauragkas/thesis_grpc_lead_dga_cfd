#!/usr/bin/env python3
"""
System Architecture & Topology Diagram Generator.
Renders a generalized, clean, and publication-grade end-to-end distributed system architecture:
  1. Distributed Storage & Spatial Indexing (LEAD DHT Cluster - N Peer Nodes)
  2. Observability & Telemetry Bus (Structured JSON logs, Kafka Event Stream, Monitor Dashboard)
  3. Island-Model Distributed Genetic Algorithm:
     - Island 1 .. Island M peer topology with P2P Elite Migration Ring
     - Each Island hosting an Orchestrator (GA Engine + Gene Cache)
     - Tier 2 Surrogate Node (MLP / Gaussian Process with Online Retraining)
     - Envoy Load Balancer (Least-Request / Power of Two Choices gRPC proxy)
     - Scalable CFD Worker Pool (W_1 .. W_K parallel AeroSandbox VLM simulators)
  4. Precise, hand-tuned geometry guaranteeing ZERO overlaps between text, boxes, and arrows.

Generates:
  - evaluation/figures/fig5_system_architecture.png
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
})


def draw_system_architecture(out_dir: Path):
    fig, ax = plt.subplots(figsize=(16.0, 10.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle="", color="#f8fafc", edge="#64748b", lw=1.3, rx=1.2, zorder=3):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.25,rounding_size={rx}",
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            zorder=zorder,
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w / 2, y + h - 2.5, title, ha="center", va="center", fontsize=8.6, fontweight="bold", color="#0f172a", zorder=zorder+1)
            ax.text(x + w / 2, y + (h - 2.5) / 2, subtitle, ha="center", va="center", fontsize=7.2, color="#334155", linespacing=1.25, zorder=zorder+1)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0f172a", zorder=zorder+1)
        return box

    def draw_arrow(x1, y1, x2, y2, label="", color="#2563eb", lw=1.4, ls="-", rad=0.0, label_pos=(0.5, 0.5), label_offset=(0, 0)):
        connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=connectionstyle,
            arrowstyle="->,head_width=3.2,head_length=4.5",
            color=color,
            linewidth=lw,
            linestyle=ls,
            zorder=5,
        )
        ax.add_patch(arrow)
        if label:
            mx = x1 + (x2 - x1) * label_pos[0] + label_offset[0]
            my = y1 + (y2 - y1) * label_pos[1] + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center", fontsize=6.8, fontweight="bold", color=color, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="none", alpha=0.92))

    # Outer network container: Isolated LAN
    lan_box = patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.6,rounding_size=2.0",
                                     facecolor="#fcfcfd", edgecolor="#94a3b8", linewidth=1.4, zorder=1)
    ax.add_patch(lan_box)
    ax.text(50, 96.8, "Isolated High-Speed Cluster Network (simulated-lan / Overlay Interconnect)",
            ha="center", fontsize=10.5, fontweight="bold", color="#1e293b")

    # =========================================================================
    # Zone 1: Distributed Storage & Spatial DHT Cluster (Top Left: x=3.5..48.5, y=55..93)
    # =========================================================================
    lead_zone = patches.FancyBboxPatch((3.5, 55), 45, 38, boxstyle="round,pad=0.6,rounding_size=1.8",
                                      facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.4, linestyle="--", zorder=2)
    ax.add_patch(lead_zone)
    ax.text(26, 90.8, "LEAD Distributed Learned DHT Ring", ha="center", fontsize=9.8, fontweight="bold", color="#1e40af")

    draw_box(5.5, 74.5, 18, 13.5, "LEAD Node 1 (Bootstrap)", "HTTP API | gRPC: 50051\nVirtual Nodes (vnodes)\nLearned RMI & Sled Storage", color="#dbeafe", edge="#3b82f6")
    draw_box(28.5, 74.5, 18, 13.5, "LEAD Node 2", "HTTP API | gRPC: 50052\nVirtual Nodes (vnodes)\nOnline PID Anchor Tuner", color="#ffffff", edge="#60a5fa")
    draw_box(17.0, 57.5, 18, 13.5, "LEAD Node N (Peer)", "HTTP API | gRPC: 50053\nVirtual Nodes (vnodes)\nPeriodic FedAvg Model Sync", color="#ffffff", edge="#60a5fa")

    # DHT Ring Arrows
    draw_arrow(23.5, 81.2, 28.5, 81.2, "Chord & FedAvg", color="#2563eb", label_offset=(0, 1.4))
    draw_arrow(37.5, 74.5, 31.0, 71.0, "", color="#2563eb")
    draw_arrow(17.0, 68.0, 11.5, 74.5, "Ring Sync", color="#2563eb", label_offset=(-2.0, -1.0))

    # =========================================================================
    # Zone 2: Telemetry, Observability & Streaming (Top Right: x=51.5..96.5, y=55..93)
    # =========================================================================
    infra_zone = patches.FancyBboxPatch((51.5, 55), 45, 38, boxstyle="round,pad=0.6,rounding_size=1.8",
                                        facecolor="#fdf4ff", edgecolor="#f0abfc", linewidth=1.4, linestyle="--", zorder=2)
    ax.add_patch(infra_zone)
    ax.text(74, 90.8, "Cluster Telemetry & Event Streaming Bus", ha="center", fontsize=9.8, fontweight="bold", color="#86198f")

    draw_box(53.5, 74.5, 18, 13.5, "Fluent-Bit Collector", "Tails: /var/log/app/*.log\nStructured JSON Ingestion\nTag-based Topic Routing", color="#ffffff", edge="#c084fc")
    draw_box(76.5, 74.5, 18, 13.5, "Apache Kafka (KRaft)", "Port: 9092 | Event Pipeline\nTopics: logs.{lead, orch,\nworker, surrogate}", color="#fae8ff", edge="#a855f7", lw=1.6)
    draw_box(63.0, 57.5, 22, 13.5, "Monitor Web Dashboard", "Node.js / Express / WebSockets\nPort: 3000 | Live GA Console\nReal-Time 3D Airframe Viewer", color="#ffffff", edge="#c084fc")

    draw_arrow(71.5, 81.2, 76.5, 81.2, "Stream", color="#9333ea", label_offset=(0, 1.4))
    draw_arrow(85.5, 74.5, 81.0, 71.0, "Events", color="#9333ea", label_offset=(1.5, 1.0))

    # =========================================================================
    # Zone 3: GA Island 1 (Bottom Left: x=3.5..48.5, y=3..48)
    # =========================================================================
    isl1_zone = patches.FancyBboxPatch((3.5, 3), 45, 45, boxstyle="round,pad=0.6,rounding_size=1.8",
                                       facecolor="#fefce8", edgecolor="#fef08a", linewidth=1.4, linestyle="--", zorder=2)
    ax.add_patch(isl1_zone)
    ax.text(26, 45.8, "GA Island 1 (Sub-Population Evolution)", ha="center", fontsize=9.8, fontweight="bold", color="#854d0e")

    # Orchestrator 1 (y=24.5..37.5) - lowered so y=37.5..48 is completely open
    draw_box(5.5, 24.5, 19, 13.0, "Orchestrator 1 (GA)", "Rust Engine | (μ + λ) Loop\nIn-Memory GeneStore Cache\nRing Port: 50060", color="#fef9c3", edge="#eab308", lw=1.6)

    # Dedicated Surrogate Node 1
    draw_box(27.5, 24.5, 19, 13.0, "Surrogate Node 1", "Rust + C++/CUDA Engine\nPort: 50054 | MLP & Matérn GP\nSliding Window Retraining", color="#ffffff", edge="#ca8a04")

    # Envoy Load Balancer 1
    draw_box(5.5, 12.0, 19, 10.0, "Envoy Proxy 1", "gRPC Proxy: 50051\nLeast-Request (P2C)\nDynamic Health Probing", color="#ffffff", edge="#eab308")

    # Worker Pool 1
    draw_box(27.5, 4.5, 19, 17.5, "CFD Worker Pool 1\n(worker = 1 .. W)", "Python + AeroSandbox\n3D Vortex Lattice (VLM)\nTrim, Lift/Drag & Stability\nHorizontally Scalable", color="#ecfdf5", edge="#10b981", lw=1.4)

    # Internal Island 1 flows
    draw_arrow(24.5, 31.0, 27.5, 31.0, "Tier 2", color="#b45309", label_offset=(0, 1.2))
    draw_arrow(15.0, 24.5, 15.0, 22.0, "Tier 3", color="#b45309", label_offset=(-1.8, 0))
    draw_arrow(24.5, 15.0, 27.5, 13.5, "gRPC", color="#059669", label_offset=(0, 1.1))

    # =========================================================================
    # Zone 4: GA Island M (Bottom Right: x=51.5..96.5, y=3..48)
    # =========================================================================
    isl2_zone = patches.FancyBboxPatch((51.5, 3), 45, 45, boxstyle="round,pad=0.6,rounding_size=1.8",
                                       facecolor="#f0fdf4", edgecolor="#bbf7d0", linewidth=1.4, linestyle="--", zorder=2)
    ax.add_patch(isl2_zone)
    ax.text(74, 45.8, "GA Island M (Peer Island Instance)", ha="center", fontsize=9.8, fontweight="bold", color="#166534")

    # Orchestrator M (y=24.5..37.5)
    draw_box(53.5, 24.5, 19, 13.0, "Orchestrator M (GA)", "Rust Engine | Diverse GA Seed\nP2P Ring Member (Port 50060)\nGeneStore & TTL Evictor", color="#dcfce7", edge="#22c55e", lw=1.6)

    # Surrogate / Peer Service
    draw_box(75.5, 24.5, 19, 13.0, "Surrogate Node M / Local", "Dedicated / Shared Predictor\nLocal Sliding Window Buffer\nSub-Millisecond Inference", color="#ffffff", edge="#16a34a")

    # Envoy Load Balancer M
    draw_box(53.5, 12.0, 19, 10.0, "Envoy Proxy M", "gRPC Least-Request Proxy\nDynamic Worker Discovery\nOutlier Detection Ejection", color="#ffffff", edge="#16a34a")

    # Worker Pool M
    draw_box(75.5, 4.5, 19, 17.5, "CFD Worker Pool M\n(worker_M = 1 .. W)", "Python + AeroSandbox\n3D Vortex Lattice (VLM)\nAutonomous Evaluation\nHorizontally Scalable", color="#ecfdf5", edge="#10b981", lw=1.4)

    # Internal Island M flows
    draw_arrow(72.5, 31.0, 75.5, 31.0, "Tier 2", color="#15803d", label_offset=(0, 1.2))
    draw_arrow(63.0, 24.5, 63.0, 22.0, "Tier 3", color="#15803d", label_offset=(-1.8, 0))
    draw_arrow(72.5, 15.0, 75.5, 13.5, "gRPC", color="#059669", label_offset=(0, 1.1))

    # =========================================================================
    # Cross-Zone Interconnects (Completely separated corridors)
    # =========================================================================
    # 1. P2P Elite Migration Ring: wide open corridor at y=41.5 and y=38.5
    draw_arrow(24.5, 41.5, 53.5, 41.5, "P2P Elite Migration (Ring gRPC | τ=5, Mc=3)", color="#dc2626", lw=1.8, label_offset=(0, 1.3))
    draw_arrow(53.5, 38.5, 24.5, 38.5, "Successor Ring Return Channel", color="#dc2626", lw=1.4, ls="--", label_offset=(0, -1.3))

    # 2. Island 1 -> LEAD DHT: vertical channel on far-left corridor (x=10.0)
    draw_arrow(10.0, 37.5, 10.0, 55.0, "Tier 1 Cache / KNN RangeQuery", color="#2563eb", lw=1.5, label_offset=(-5.5, 0))

    # 3. Island M -> LEAD DHT: dedicated angle into bottom of LEAD zone (x=30.0, y=55.0)
    draw_arrow(58.0, 37.5, 34.0, 55.0, "Spatial Key Ingestion (PutRouted)", color="#2563eb", lw=1.4, ls="--", label_offset=(3.0, 1.2))

    # 4. Telemetry Stream to Fluent-Bit: clear bridge between zones at y=79.0
    draw_arrow(48.5, 79.0, 53.5, 79.0, "Log Volume", color="#9333ea", lw=1.4, ls=":", label_offset=(0, 1.3))

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
