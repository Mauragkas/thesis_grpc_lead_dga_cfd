#!/usr/bin/env python3
"""
Plots Criterion.rs benchmark results comparing CPU OpenMP vs CUDA GPU
and scaling behavior for Gaussian Process linear algebra operations.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Subplot 1: Covariance Matrix Computation Scaling
    ax1 = axes[0]
    ax1.plot(n_samples, cpu_cov_ms, "o-", color="#2563eb", lw=2, label="Host CPU (OpenMP 12t)")
    ax1.plot(n_samples, cuda_cov_ms, "s--", color="#10b981", lw=2, label="NVIDIA CUDA GPU")
    ax1.set_xlabel("Number of Samples (N)", fontsize=12)
    ax1.set_ylabel("Execution Time (ms)", fontsize=12)
    ax1.set_title("Kernel Covariance Matrix Scaling (D=11)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=11)

    # Subplot 2: Cholesky Decomposition Scaling
    ax2 = axes[1]
    ax2.plot(n_samples, cholesky_ms, "D-", color="#8b5cf6", lw=2, label="Cholesky Factorization L")
    # Add theoretical O(N^3) fit curve
    poly_fit = np.polyfit(n_samples**3, cholesky_ms, 1)
    n_smooth = np.linspace(100, 1000, 100)
    ax2.plot(n_smooth, poly_fit[0] * (n_smooth**3) + poly_fit[1], "k:", alpha=0.5, label="O(N³) Theoretical")
    ax2.set_xlabel("Matrix Dimension (N x N)", fontsize=12)
    ax2.set_ylabel("Execution Time (ms)", fontsize=12)
    ax2.set_title("Cholesky Factorization L Lᵀ = K", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=11)

    # Subplot 3: Batch Inference Latency & Throughput
    ax3 = axes[2]
    ax3.plot(m_test, inference_ms, "^-", color="#f59e0b", lw=2, label="Prediction Latency")
    ax3.set_xlabel("Test Batch Size (M)", fontsize=12)
    ax3.set_ylabel("Total Latency (ms)", fontsize=12)
    ax3.set_title("Batch Inference Scaling (N_train=360)", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    # Secondary y-axis for throughput (samples/sec)
    ax3_twin = ax3.twinx()
    throughput = (m_test / (inference_ms * 1e-3))
    ax3_twin.plot(m_test, throughput, "r--", alpha=0.7, label="Throughput (evals/s)")
    ax3_twin.set_ylabel("Throughput (designs/sec)", color="r", fontsize=12)
    ax3_twin.tick_params(axis="y", labelcolor="r")

    save_path = Path(__file__).parent / "benchmark_results.png"
    plt.savefig(save_path, dpi=200)
    print(f"Benchmark plot successfully generated at: {save_path}")

if __name__ == "__main__":
    main()
