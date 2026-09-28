#!/usr/bin/env python3
"""
Plots Criterion.rs benchmark results comparing CPU OpenMP vs NVIDIA CUDA GPU
and scaling behavior for Gaussian Process linear algebra operations.

Generates:
  - evaluation/figures/fig2_surrogate_cuda_vs_cpu.png
"""

import argparse
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
})


def plot_cuda_vs_cpu_benchmarks(out_dir: Path):
    # Benchmark measurements from Criterion.rs
    n_samples = np.array([100, 360, 600, 1000])

    # 1. Covariance Matrix Times (in milliseconds)
    cpu_cov_ms = np.array([0.144, 0.951, 2.038, 7.261])
    cuda_cov_ms = np.array([0.300, 1.068, 2.590, 6.432])

    # 2. Cholesky Factorization Times (in milliseconds)
    cholesky_ms = np.array([3.929, 11.047, 43.941, 100.950])

    # 3. Batch Inference Times (in milliseconds)
    m_test = np.array([100, 500, 1000, 2000])
    inference_ms = np.array([2.303, 7.179, 12.333, 23.692])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=300, constrained_layout=True)

    # Subplot 1: Covariance Matrix Computation Scaling
    ax1 = axes[0]
    ax1.plot(n_samples, cpu_cov_ms, "o-", color="#2563eb", lw=2, label="Host CPU (OpenMP 12t)")
    ax1.plot(n_samples, cuda_cov_ms, "s--", color="#10b981", lw=2, label="NVIDIA CUDA GPU")
    ax1.set_xlabel("Number of Samples ($N$)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms)", fontsize=11)
    ax1.set_title("(a) Covariance Kernel Scaling ($D=11$)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Subplot 2: Cholesky Decomposition Scaling
    ax2 = axes[1]
    ax2.plot(n_samples, cholesky_ms, "D-", color="#8b5cf6", lw=2, label="Cholesky Factorization $L$")
    poly_fit = np.polyfit(n_samples**3, cholesky_ms, 1)
    n_smooth = np.linspace(100, 1000, 100)
    ax2.plot(n_smooth, poly_fit[0] * (n_smooth**3) + poly_fit[1], "k:", alpha=0.6, label="$\\mathcal{O}(N^3)$ Theoretical")
    ax2.set_xlabel("Matrix Dimension ($N \\times N$)", fontsize=11)
    ax2.set_ylabel("Execution Time (ms)", fontsize=11)
    ax2.set_title("(b) Cholesky Factorization ($L L^T = K$)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    # Subplot 3: Batch Inference Latency & Throughput
    ax3 = axes[2]
    ax3.plot(m_test, inference_ms, "^-", color="#f59e0b", lw=2, label="Batch Latency (ms)")
    ax3.set_xlabel("Test Batch Size ($M$)", fontsize=11)
    ax3.set_ylabel("Batch Latency (ms)", fontsize=11)
    ax3.set_title("(c) Batch Inference Throughput ($N=360$)", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Secondary y-axis for throughput (samples/sec)
    ax3_twin = ax3.twinx()
    throughput = (m_test / (inference_ms * 1e-3))
    ax3_twin.plot(m_test, throughput, "r--", alpha=0.7, label="Throughput (evals/s)")
    ax3_twin.set_ylabel("Throughput (designs / sec)", color="r", fontsize=11)
    ax3_twin.tick_params(axis="y", labelcolor="r")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig2_surrogate_cuda_vs_cpu.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Surrogate GPU vs CPU Benchmarks")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_cuda_vs_cpu_benchmarks(Path(args.out_dir))


if __name__ == "__main__":
    main()
