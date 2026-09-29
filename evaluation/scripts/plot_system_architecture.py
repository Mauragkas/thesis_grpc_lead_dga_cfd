#!/usr/bin/env python3
"""
System Architecture & Topology Diagram Generator.
Renders the complete end-to-end distributed system architecture matching the
multi-island simulated LAN deployment defined in compose/toRun.md:
  1. Observability & Event Streaming (docker-compose.infra.yml):
     - Fluent-Bit (log tailing from shared app-logs volume)
     - Apache Kafka (KRaft mode broker, port 9092)
     - Node.js WebSocket Monitor Dashboard (port 3000)
  2. LEAD Distributed DHT Ring (docker-compose.lead.yml):
     - lead-node1 (Seed, port 2001/50051)
     - lead-node2 (port 2002/50052)
     - lead-node3 (port 2003/50053)
     - Chord ring topology & FedAvg model synchronization
  3. Island 1 (docker-compose.worker.yml):
     - Orchestrator 1 (GA Seed 42, ring port 50060)
     - Surrogate Node (port 50054, MLP / GP, online retraining)
     - Envoy Load Balancer 1 (port 50051, Least-Request gRPC)
     - Worker Pool 1 (--scale worker=4, AeroSandbox VLM CFD)
  4. Island 2 (docker-compose.worker2.yml):
     - Orchestrator 2 (GA Seed 43, joins Island 1 ring port 50060)
     - Envoy Load Balancer 2 (port 50052 -> 50051)
     - Worker Pool 2 (--scale worker2=4, AeroSandbox VLM CFD)
  5. Cross-Subsystem Interconnects:
     - Asynchronous ring migration between Orchestrator 1 and 2
     - Spatial k-NN / PutRouted lookups into LEAD DHT
     - JSON structured logging pipeline into Fluent-Bit

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
    "font.size": 9.5,
    "figure.autolayout": True,
})


def draw_system_architecture(out_dir: Path):
    fig, ax = plt.subplots(figsize=(15.0, 9.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Helper styling function
    def draw_box(x, y, w, h, title, subtitle="", color="#f8fafc", edge="#64748b", lw=1.4, rx=1.2):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.2,rounding_size={rx}",
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            zorder=3,
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w / 2, y + h - 2.5, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0f172a", zorder=4)
            ax.text(x + w / 2, y + (h - 2.5) / 2, subtitle, ha="center", va="center", fontsize=7.2, color="#334155", zorder=4)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0f172a", zorder=4)
        return box

    def draw_arrow(x1, y1, x2, y2, label="", color="#2563eb", lw=1.5, ls="-", rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=connectionstyle,
            arrowstyle="->,head_width=3.2,head_length=4.6",
            color=color,
            linewidth=lw,
            linestyle=ls,
            zorder=5,
        )
        ax.add_patch(arrow)
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + 1.2
            ax.text(mx, my, label, ha="center", va="center", fontsize=6.8, fontweight="bold", color=color, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#ffffff", edgecolor="none", alpha=0.9))

    # Outer network container: simulated-lan
    lan_box = patches.FancyBboxPatch((1, 1), 98, 98, boxstyle="round,pad=0.6,rounding_size=2.0",
                                     facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5, zorder=1)
    ax.add_patch(lan_box)
    ax.text(50, 97.2, "Simulated LAN Environment (External Docker Network: simulated-lan | Shared Volume: app-logs)",
            ha="center", fontsize=10.5, fontweight="bold", color="#334155")

    # =========================================================================
    # Block 1: LEAD Distributed DHT Ring (Top Left: x=3..48, y=56..94)
    # =========================================================================
    lead_zone = patches.FancyBboxPatch((3, 56), 46, 38, boxstyle="round,pad=0.6,rounding_size=1.8",
                                      facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.5, linestyle="--", zorder=2)
    ax.add_patch(lead_zone)
    ax.text(26, 91.8, "LEAD Distributed DHT Cluster (docker-compose.lead.yml)",
            ha="center", fontsize=9.8, fontweight="bold", color="#1e40af")

    # 3 LEAD Peer Nodes
    draw_box(5, 74, 18, 14, "lead-node1 (Seed)", "HTTP: 2001 | gRPC: 50051\n100 vnodes | Sled Storage\nLearned Index RMI", color="#dbeafe", edge="#3b82f6")
    draw_box(29, 74, 18, 14, "lead-node2", "HTTP: 2002 | gRPC: 50052\n100 vnodes | Sled Storage\nJoins lead-node1", color="#ffffff", edge="#60a5fa")
    draw_box(17, 58, 18, 13, "lead-node3", "HTTP: 2003 | gRPC: 50053\n100 vnodes | Sled Storage\nJoins lead-node1", color="#ffffff", edge="#60a5fa")

    # Internal DHT ring & FedAvg sync arrows
    draw_arrow(23, 81, 29, 81, "Chord & FedAvg", color="#2563eb")
    draw_arrow(38, 74, 30, 68, "", color="#2563eb", rad=0.1)
    draw_arrow(22, 68, 14, 74, "Chord Ring", color="#2563eb", rad=0.1)

    # =========================================================================
    # Block 2: Observability & Telemetry (Top Right: x=51..97, y=56..94)
    # =========================================================================
    infra_zone = patches.FancyBboxPatch((51, 56), 46, 38, boxstyle="round,pad=0.6,rounding_size=1.8",
                                        facecolor="#fdf4ff", edgecolor="#f0abfc", linewidth=1.5, linestyle="--", zorder=2)
    ax.add_patch(infra_zone)
    ax.text(74, 91.8, "Observability & Event Streaming (docker-compose.infra.yml)",
            ha="center", fontsize=9.8, fontweight="bold", color="#86198f")

    draw_box(53, 74, 19, 14, "Fluent-Bit 3.0", "Tails: /var/log/app/*.log\nPort 24224 (TCP input)\nRoutes tags by role", color="#ffffff", edge="#c084fc")
    draw_box(76, 74, 19, 14, "Apache Kafka 3.9", "KRaft Mode (Broker: 9092)\nTopics: logs.{lead, orch,\nworker, surrogate}", color="#fae8ff", edge="#a855f7", lw=1.8)
    draw_box(63, 58, 22, 13, "Monitor Web UI", "Node.js / Express / WS\nPort: 3000 | Live GA Console\n3D Aerodynamic Viewer", color="#ffffff", edge="#c084fc")

    draw_arrow(72, 81, 76, 81, "JSON logs", color="#9333ea")
    draw_arrow(85, 74, 79, 68, "Kafka Stream", color="#9333ea", rad=-0.1)

    # =========================================================================
    # Block 3: GA Island 1 (Bottom Left: x=3..48, y=4..52)
    # =========================================================================
    isl1_zone = patches.FancyBboxPatch((3, 4), 46, 48, boxstyle="round,pad=0.6,rounding_size=1.8",
                                       facecolor="#fefce8", edgecolor="#fef08a", linewidth=1.5, linestyle="--", zorder=2)
    ax.add_patch(isl1_zone)
    ax.text(26, 49.5, "GA Island 1 (docker-compose.worker.yml)",
            ha="center", fontsize=9.8, fontweight="bold", color="#854d0e")

    # Orchestrator 1
    draw_box(5, 33, 20, 13, "Orchestrator 1", "Rust GA Seed: 42\nRing Port: 50060\nGeneStore & Evictor", color="#fef9c3", edge="#eab308", lw=1.8)

    # Surrogate Node
    draw_box(27, 33, 20, 13, "surrogate-node 1", "Rust + C++/CUDA Engine\nPort: 50054 | MLP + GP\nSliding Window Buffer", color="#ffffff", edge="#ca8a04")

    # Envoy Load Balancer 1
    draw_box(5, 17, 20, 11, "Envoy Proxy 1", "Port: 50051 (gRPC)\nLeast-Request / P2C\nDynamic Discovery", color="#ffffff", edge="#eab308")

    # Worker Pool 1
    draw_box(27, 7, 20, 21, "Worker Pool 1\n(--scale worker=4)", "Python + AeroSandbox\n3D Vortex Lattice (VLM)\nTrim & Static Stability\nworker-1 .. worker-4", color="#ecfdf5", edge="#10b981", lw=1.5)

    # Internal Island 1 flows
    draw_arrow(25, 40, 27, 40, "Tier 2 Predict", color="#b45309")
    draw_arrow(15, 33, 15, 28, "Tier 3 Batch", color="#b45309")
    draw_arrow(25, 22, 27, 20, "Round-Robin gRPC", color="#059669")

    # =========================================================================
    # Block 4: GA Island 2 (Bottom Right: x=51..97, y=4..52)
    # =========================================================================
    isl2_zone = patches.FancyBboxPatch((51, 4), 46, 48, boxstyle="round,pad=0.6,rounding_size=1.8",
                                       facecolor="#f0fdf4", edgecolor="#bbf7d0", linewidth=1.5, linestyle="--", zorder=2)
    ax.add_patch(isl2_zone)
    ax.text(74, 49.5, "GA Island 2 (docker-compose.worker2.yml)",
            ha="center", fontsize=9.8, fontweight="bold", color="#166534")

    # Orchestrator 2
    draw_box(53, 33, 20, 13, "Orchestrator 2", "Rust GA Seed: 43\nRing: 50060 (Joins Orch 1)\nGeneStore & Evictor", color="#dcfce7", edge="#22c55e", lw=1.8)

    # Envoy Load Balancer 2
    draw_box(75, 33, 20, 13, "Envoy Proxy 2", "Host: 50052 -> 50051\nLeast-Request / P2C\nDynamic Discovery", color="#ffffff", edge="#16a34a")

    # Worker Pool 2
    draw_box(75, 7, 20, 21, "Worker Pool 2\n(--scale worker2=4)", "Python + AeroSandbox\n3D Vortex Lattice (VLM)\nTrim & Static Stability\nworker2-1 .. worker2-4", color="#ecfdf5", edge="#10b981", lw=1.5)

    # Internal Island 2 flows
    draw_arrow(73, 40, 75, 40, "EvaluateBatch", color="#15803d")
    draw_arrow(85, 33, 85, 28, "Round-Robin gRPC", color="#059669")

    # =========================================================================
    # Cross-System Communications
    # =========================================================================
    # 1. Island Migration Ring between Orchestrator 1 and Orchestrator 2
    draw_arrow(25, 44, 53, 44, "P2P Island Migration (Ring gRPC: 50060 | τ=5, Mc=3)", color="#dc2626", lw=2.0)
    draw_arrow(53, 38, 25, 38, "Top-K Elites Exchange", color="#dc2626", lw=1.6)

    # 2. Orchestrator -> LEAD DHT lookups / insertions
    draw_arrow(12, 46, 12, 58, "Tier 1 Put / KNN RangeQuery", color="#2563eb", lw=1.6, ls="-")
    draw_arrow(56, 46, 32, 60, "PutRouted (DHT Cache)", color="#2563eb", lw=1.4, ls="--", rad=0.1)

    # 3. Logging into shared volume / Fluent-Bit
    draw_arrow(49, 74, 53, 78, "Shared Volume /var/log/app (app-logs)", color="#9333ea", lw=1.5, ls=":")

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
