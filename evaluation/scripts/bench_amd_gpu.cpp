#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <omp.h>

// ============================================================================
// Statistical Helper Functions (10 Trials -> Mean, Std, Var, Min, Max)
// ============================================================================
struct Stats {
    double mean;
    double std;
    double var;
    double min_val;
    double max_val;
};

Stats compute_stats(const std::vector<double>& samples) {
    if (samples.empty()) return {0.0, 0.0, 0.0, 0.0, 0.0};
    double sum = 0.0;
    double min_v = samples[0];
    double max_v = samples[0];
    for (double v : samples) {
        sum += v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    double mean = sum / samples.size();
    double var = 0.0;
    for (double v : samples) {
        double diff = v - mean;
        var += diff * diff;
    }
    if (samples.size() > 1) {
        var /= (samples.size() - 1); // Sample variance
    } else {
        var = 0.0;
    }
    double std_dev = std::sqrt(var);
    return {mean, std_dev, var, min_v, max_v};
}

// ============================================================================
// 1. GAUSSIAN PROCESS Matérn 5/2 Covariance Kernels
// ============================================================================
__device__ inline double d_eval_matern52(double d2, double sig_var) {
    double dist = sqrt(d2 > 0.0 ? d2 : 0.0);
    double sqrt5_r = 2.23606797749979 * dist;
    return sig_var * (1.0 + sqrt5_r + (5.0 / 3.0) * d2) * exp(-sqrt5_r);
}

inline double cpu_eval_matern52(double d2, double sig_var) {
    double dist = sqrt(d2 > 0.0 ? d2 : 0.0);
    double sqrt5_r = 2.23606797749979 * dist;
    return sig_var * (1.0 + sqrt5_r + (5.0 / 3.0) * d2) * exp(-sqrt5_r);
}

__global__ void k_compute_covariance(
    const double* __restrict__ X1, size_t n1,
    const double* __restrict__ X2, size_t n2,
    size_t dim, const double* __restrict__ ls,
    double sig_var, double noise_var,
    bool add_noise, bool is_symmetric,
    double* __restrict__ out_K
) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n1 && j < n2) {
        if (is_symmetric && j < i) return;

        double sum = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            double diff = (X1[i * dim + d] - X2[j * dim + d]) / ls[d];
            sum += diff * diff;
        }

        double val = d_eval_matern52(sum, sig_var);
        if (add_noise && i == j) val += noise_var;

        out_K[i * n2 + j] = val;
        if (is_symmetric && i != j) out_K[j * n1 + i] = val;
    }
}

void cpu_compute_covariance(
    const double* X1, size_t n1,
    const double* X2, size_t n2,
    size_t dim, const double* ls,
    double sig_var, double noise_var,
    bool add_noise, bool is_symmetric,
    double* out_K
) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n1; ++i) {
        size_t start_j = is_symmetric ? i : 0;
        for (size_t j = start_j; j < n2; ++j) {
            double sum = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                double diff = (X1[i * dim + d] - X2[j * dim + d]) / ls[d];
                sum += diff * diff;
            }
            double val = cpu_eval_matern52(sum, sig_var);
            if (add_noise && i == j) val += noise_var;
            out_K[i * n2 + j] = val;
            if (is_symmetric && i != j) {
                out_K[j * n1 + i] = val;
            }
        }
    }
}

// Cholesky decomposition L * L^T = A
void cpu_cholesky(size_t n, const double* A, double* L) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                double val = A[i * n + i] - sum;
                L[i * n + j] = sqrt(val > 1e-12 ? val : 1e-12);
            } else {
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
}

// ============================================================================
// 2. NEURAL NETWORK MLP BATCH INFERENCE (11 -> 64 -> 32 -> 1)
// ============================================================================
__device__ inline double dev_relu(double x) {
    return x > 0.0 ? x : 0.0;
}

__global__ void k_mlp_predict(
    const double* __restrict__ d_x_test, size_t n_test, size_t in_dim,
    const double* __restrict__ d_w0, const double* __restrict__ d_b0, size_t h0_dim,
    const double* __restrict__ d_w1, const double* __restrict__ d_b1, size_t h1_dim,
    const double* __restrict__ d_w_out, const double* __restrict__ d_b_out,
    double* __restrict__ d_out_pred
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_test) return;

    const double* x = &d_x_test[idx * in_dim];
    double h0[128];
    for (size_t j = 0; j < h0_dim; ++j) {
        double sum = d_b0[j];
        for (size_t i = 0; i < in_dim; ++i) {
            sum += x[i] * d_w0[i * h0_dim + j];
        }
        h0[j] = dev_relu(sum);
    }

    double h1[128];
    for (size_t j = 0; j < h1_dim; ++j) {
        double sum = d_b1[j];
        for (size_t i = 0; i < h0_dim; ++i) {
            sum += h0[i] * d_w1[i * h1_dim + j];
        }
        h1[j] = dev_relu(sum);
    }

    double sum = d_b_out[0];
    for (size_t i = 0; i < h1_dim; ++i) {
        sum += h1[i] * d_w_out[i];
    }
    d_out_pred[idx] = sum;
}

void cpu_mlp_predict(
    const double* x_test, size_t n_test, size_t in_dim,
    const double* w0, const double* b0, size_t h0_dim,
    const double* w1, const double* b1, size_t h1_dim,
    const double* w_out, const double* b_out,
    double* out_pred
) {
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < n_test; ++t) {
        const double* x = &x_test[t * in_dim];
        double h0[128];
        for (size_t j = 0; j < h0_dim; ++j) {
            double sum = b0[j];
            for (size_t i = 0; i < in_dim; ++i) sum += x[i] * w0[i * h0_dim + j];
            h0[j] = sum > 0.0 ? sum : 0.0;
        }

        double h1[128];
        for (size_t j = 0; j < h1_dim; ++j) {
            double sum = b1[j];
            for (size_t i = 0; i < h0_dim; ++i) sum += h0[i] * w1[i * h1_dim + j];
            h1[j] = sum > 0.0 ? sum : 0.0;
        }

        double sum = b_out[0];
        for (size_t i = 0; i < h1_dim; ++i) sum += h1[i] * w_out[i];
        out_pred[t] = sum;
    }
}

// ============================================================================
// MAIN BENCHMARK DRIVER (10 TRIALS PER CONFIGURATION)
// ============================================================================
int main(int argc, char* argv[]) {
    std::string json_output = "evaluation/data/amd_gpu_benchmarks.json";
    int num_trials = 10;
    if (argc > 1) {
        json_output = argv[1];
    }
    if (argc > 2) {
        num_trials = std::atoi(argv[2]);
        if (num_trials < 3) num_trials = 3;
    }

    (void)hipSetDevice(0);
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
        std::cerr << "Failed to find HIP/ROCm device 0: " << hipGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << " AMD ROCm / HIP STATISTICAL BENCHMARKS (" << num_trials << " TRIALS PER CONFIGURATION)\n";
    std::cout << " Device: " << prop.name << " (Architecture: gfx1102 / RDNA 3)\n";
    std::cout << " Compute Units: " << prop.multiProcessorCount << " | Total VRAM: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << " Host Threads:  " << omp_get_max_threads() << " OpenMP workers\n";
    std::cout << "================================================================================\n\n";

    // ------------------------------------------------------------------------
    // Global Warmup to Lock GPU Frequency & Power States
    // ------------------------------------------------------------------------
    std::cout << "[*] Executing GPU warmup to stabilize DPM power states & clocks...\n";
    {
        const size_t w_n = 256, w_dim = 11;
        std::vector<double> w_x(w_n * w_dim, 0.5), w_k(w_n * w_n, 0.0), w_ls(w_dim, 1.0);
        double *dw_x, *dw_k, *dw_ls;
        (void)hipMalloc(&dw_x, w_x.size() * sizeof(double));
        (void)hipMalloc(&dw_k, w_k.size() * sizeof(double));
        (void)hipMalloc(&dw_ls, w_ls.size() * sizeof(double));
        (void)hipMemcpy(dw_x, w_x.data(), w_x.size() * sizeof(double), hipMemcpyHostToDevice);
        (void)hipMemcpy(dw_ls, w_ls.data(), w_ls.size() * sizeof(double), hipMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((w_n + 15) / 16, (w_n + 15) / 16);
        for (int i = 0; i < 50; ++i) {
            k_compute_covariance<<<grid, block>>>(dw_x, w_n, dw_x, w_n, w_dim, dw_ls, 1.5, 1e-3, true, true, dw_k);
        }
        (void)hipDeviceSynchronize();
        (void)hipFree(dw_x); (void)hipFree(dw_k); (void)hipFree(dw_ls);
    }
    std::cout << "[✓] GPU Warmup completed.\n\n";

    // ------------------------------------------------------------------------
    // 1. GP Covariance Benchmark (N = 100, 360, 600, 1000)
    // ------------------------------------------------------------------------
    struct GPStatResult {
        size_t n;
        Stats cpu_cov;
        Stats rocm_cov;
        Stats cholesky;
        double speedup;
    };
    std::vector<GPStatResult> gp_stat_results;

    const size_t dim = 11;
    std::vector<double> h_ls(dim, 1.0);
    double *d_ls;
    (void)hipMalloc(&d_ls, dim * sizeof(double));
    (void)hipMemcpy(d_ls, h_ls.data(), dim * sizeof(double), hipMemcpyHostToDevice);

    std::vector<size_t> N_sizes = {100, 360, 600, 1000};
    std::cout << "[1/2] Benchmarking Gaussian Process Covariance & Cholesky (D=11)...\n";
    std::cout << "-----------------------------------------------------------------------------------------------------------------\n";
    std::cout << "    N |   CPU Cov (ms) [μ±σ]   |  ROCm Cov (ms) [μ±σ]  | Speedup (x) | CPU Cholesky (ms) [μ±σ]\n";
    std::cout << "-----------------------------------------------------------------------------------------------------------------\n";

    for (size_t n : N_sizes) {
        std::vector<double> h_X(n * dim, 0.5);
        std::vector<double> h_K_cpu(n * n, 0.0);
        std::vector<double> h_L(n * n, 0.0);

        double *d_X, *d_K;
        (void)hipMalloc(&d_X, n * dim * sizeof(double));
        (void)hipMalloc(&d_K, n * n * sizeof(double));
        (void)hipMemcpy(d_X, h_X.data(), n * dim * sizeof(double), hipMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (n + 15) / 16);

        // Pre-warmup for this N
        for (int i = 0; i < 20; ++i) {
            k_compute_covariance<<<grid, block>>>(d_X, n, d_X, n, dim, d_ls, 1.5, 1e-3, true, true, d_K);
        }
        (void)hipDeviceSynchronize();

        // 10 Trials for GPU Covariance
        std::vector<double> gpu_trials;
        const int reps_gpu = 50;
        for (int t = 0; t < num_trials; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < reps_gpu; ++i) {
                k_compute_covariance<<<grid, block>>>(d_X, n, d_X, n, dim, d_ls, 1.5, 1e-3, true, true, d_K);
            }
            (void)hipDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_trials.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / reps_gpu);
        }

        // 10 Trials for CPU Covariance
        std::vector<double> cpu_trials;
        const int reps_cpu = (n <= 360) ? 30 : 15;
        for (int t = 0; t < num_trials; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < reps_cpu; ++i) {
                cpu_compute_covariance(h_X.data(), n, h_X.data(), n, dim, h_ls.data(), 1.5, 1e-3, true, true, h_K_cpu.data());
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_trials.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / reps_cpu);
        }

        // 10 Trials for CPU Cholesky
        std::vector<double> chol_trials;
        const int reps_chol = (n <= 360) ? 20 : 5;
        for (int t = 0; t < num_trials; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < reps_chol; ++i) {
                cpu_cholesky(n, h_K_cpu.data(), h_L.data());
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            chol_trials.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / reps_chol);
        }

        Stats s_cpu = compute_stats(cpu_trials);
        Stats s_gpu = compute_stats(gpu_trials);
        Stats s_chol = compute_stats(chol_trials);
        double speedup = s_cpu.mean / s_gpu.mean;

        gp_stat_results.push_back({n, s_cpu, s_gpu, s_chol, speedup});

        std::cout << std::setw(5) << n << " | "
                  << std::fixed << std::setprecision(3) << std::setw(7) << s_cpu.mean << " ± " << std::setw(5) << s_cpu.std << " ms | "
                  << std::fixed << std::setprecision(3) << std::setw(7) << s_gpu.mean << " ± " << std::setw(5) << s_gpu.std << " ms | "
                  << std::fixed << std::setprecision(2) << std::setw(11) << speedup << " | "
                  << std::fixed << std::setprecision(3) << std::setw(9) << s_chol.mean << " ± " << std::setw(6) << s_chol.std << " ms\n";

        (void)hipFree(d_X);
        (void)hipFree(d_K);
    }
    (void)hipFree(d_ls);

    // ------------------------------------------------------------------------
    // 2. MLP Batch Inference Benchmark (M = 100 ... 10000)
    // ------------------------------------------------------------------------
    struct MLPStatResult {
        size_t m;
        Stats cpu;
        Stats rocm;
        double speedup;
        double cpu_throughput;
        double rocm_throughput;
    };
    std::vector<MLPStatResult> mlp_stat_results;

    size_t in_dim = 11, h0_dim = 64, h1_dim = 32;
    std::vector<double> w0(in_dim * h0_dim, 0.05), b0(h0_dim, 0.01);
    std::vector<double> w1(h0_dim * h1_dim, 0.05), b1(h1_dim, 0.01);
    std::vector<double> w_out(h1_dim, 0.05), b_out(1, 0.01);

    double *d_w0, *d_b0, *d_w1, *d_b1, *d_w_out, *d_b_out;
    (void)hipMalloc(&d_w0, w0.size() * sizeof(double));
    (void)hipMalloc(&d_b0, b0.size() * sizeof(double));
    (void)hipMalloc(&d_w1, w1.size() * sizeof(double));
    (void)hipMalloc(&d_b1, b1.size() * sizeof(double));
    (void)hipMalloc(&d_w_out, w_out.size() * sizeof(double));
    (void)hipMalloc(&d_b_out, b_out.size() * sizeof(double));

    (void)hipMemcpy(d_w0, w0.data(), w0.size() * sizeof(double), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_b0, b0.data(), b0.size() * sizeof(double), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_w1, w1.data(), w1.size() * sizeof(double), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_b1, b1.data(), b1.size() * sizeof(double), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_w_out, w_out.data(), w_out.size() * sizeof(double), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_b_out, b_out.data(), b_out.size() * sizeof(double), hipMemcpyHostToDevice);

    std::vector<size_t> M_sizes = {100, 360, 600, 1000, 2000, 5000, 10000};
    std::cout << "\n[2/2] Benchmarking Neural Network MLP Batch Inference (11 -> 64 -> 32 -> 1)...\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "     M |     CPU Lat (ms) [μ±σ]     |    ROCm Lat (ms) [μ±σ]    | Speedup (x) |  CPU Throughput  |  ROCm Throughput\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------\n";

    for (size_t m : M_sizes) {
        std::vector<double> x_test(m * in_dim, 0.5);
        std::vector<double> out_cpu(m, 0.0);
        double *d_x, *d_out;
        (void)hipMalloc(&d_x, m * in_dim * sizeof(double));
        (void)hipMalloc(&d_out, m * sizeof(double));
        (void)hipMemcpy(d_x, x_test.data(), m * in_dim * sizeof(double), hipMemcpyHostToDevice);

        int threads = 256;
        int blocks = (m + threads - 1) / threads;

        // Dedicated Warmup for this M
        for (int i = 0; i < 30; ++i) {
            k_mlp_predict<<<blocks, threads>>>(d_x, m, in_dim, d_w0, d_b0, h0_dim, d_w1, d_b1, h1_dim, d_w_out, d_b_out, d_out);
        }
        (void)hipDeviceSynchronize();

        // 10 Trials for GPU MLP
        std::vector<double> gpu_trials;
        const int reps_gpu = 100;
        for (int t = 0; t < num_trials; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < reps_gpu; ++i) {
                k_mlp_predict<<<blocks, threads>>>(d_x, m, in_dim, d_w0, d_b0, h0_dim, d_w1, d_b1, h1_dim, d_w_out, d_b_out, d_out);
            }
            (void)hipDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_trials.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / reps_gpu);
        }

        // 10 Trials for CPU MLP
        std::vector<double> cpu_trials;
        const int reps_cpu = 30;
        for (int t = 0; t < num_trials; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < reps_cpu; ++i) {
                cpu_mlp_predict(x_test.data(), m, in_dim, w0.data(), b0.data(), h0_dim, w1.data(), b1.data(), h1_dim, w_out.data(), b_out.data(), out_cpu.data());
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_trials.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / reps_cpu);
        }

        Stats s_cpu = compute_stats(cpu_trials);
        Stats s_gpu = compute_stats(gpu_trials);
        double speedup = s_cpu.mean / s_gpu.mean;
        double throughput_cpu = m / (s_cpu.mean * 1e-3);
        double throughput_gpu = m / (s_gpu.mean * 1e-3);

        mlp_stat_results.push_back({m, s_cpu, s_gpu, speedup, throughput_cpu, throughput_gpu});

        std::cout << std::setw(6) << m << " | "
                  << std::fixed << std::setprecision(4) << std::setw(8) << s_cpu.mean << " ± " << std::setw(6) << s_cpu.std << " ms | "
                  << std::fixed << std::setprecision(4) << std::setw(8) << s_gpu.mean << " ± " << std::setw(6) << s_gpu.std << " ms | "
                  << std::fixed << std::setprecision(2) << std::setw(11) << speedup << " | "
                  << std::scientific << std::setprecision(2) << std::setw(14) << throughput_cpu << " ev/s | "
                  << std::scientific << std::setprecision(2) << std::setw(14) << throughput_gpu << " ev/s\n";

        (void)hipFree(d_x);
        (void)hipFree(d_out);
    }

    (void)hipFree(d_w0); (void)hipFree(d_b0); (void)hipFree(d_w1); (void)hipFree(d_b1); (void)hipFree(d_w_out); (void)hipFree(d_b_out);

    // Save Extended Statistical JSON results
    std::ofstream out(json_output);
    if (out.is_open()) {
        out << "{\n";
        out << "  \"device\": \"" << prop.name << "\",\n";
        out << "  \"arch\": \"gfx1102\",\n";
        out << "  \"compute_units\": " << prop.multiProcessorCount << ",\n";
        out << "  \"vram_mb\": " << (prop.totalGlobalMem / (1024 * 1024)) << ",\n";
        out << "  \"num_trials\": " << num_trials << ",\n";
        out << "  \"gp_benchmarks\": [\n";
        for (size_t i = 0; i < gp_stat_results.size(); ++i) {
            const auto& r = gp_stat_results[i];
            out << "    {\n"
                << "      \"n\": " << r.n << ",\n"
                << "      \"cpu_cov_ms\": " << r.cpu_cov.mean << ",\n"
                << "      \"cpu_cov_std\": " << r.cpu_cov.std << ",\n"
                << "      \"cpu_cov_var\": " << r.cpu_cov.var << ",\n"
                << "      \"rocm_cov_ms\": " << r.rocm_cov.mean << ",\n"
                << "      \"rocm_cov_std\": " << r.rocm_cov.std << ",\n"
                << "      \"rocm_cov_var\": " << r.rocm_cov.var << ",\n"
                << "      \"speedup\": " << r.speedup << ",\n"
                << "      \"cholesky_ms\": " << r.cholesky.mean << ",\n"
                << "      \"cholesky_std\": " << r.cholesky.std << ",\n"
                << "      \"cholesky_var\": " << r.cholesky.var << "\n"
                << "    }" << (i + 1 < gp_stat_results.size() ? "," : "") << "\n";
        }
        out << "  ],\n";
        out << "  \"mlp_benchmarks\": [\n";
        for (size_t i = 0; i < mlp_stat_results.size(); ++i) {
            const auto& r = mlp_stat_results[i];
            out << "    {\n"
                << "      \"m\": " << r.m << ",\n"
                << "      \"cpu_ms\": " << r.cpu.mean << ",\n"
                << "      \"cpu_std\": " << r.cpu.std << ",\n"
                << "      \"cpu_var\": " << r.cpu.var << ",\n"
                << "      \"rocm_ms\": " << r.rocm.mean << ",\n"
                << "      \"rocm_std\": " << r.rocm.std << ",\n"
                << "      \"rocm_var\": " << r.rocm.var << ",\n"
                << "      \"speedup\": " << r.speedup << ",\n"
                << "      \"cpu_throughput\": " << r.cpu_throughput << ",\n"
                << "      \"rocm_throughput\": " << r.rocm_throughput << "\n"
                << "    }" << (i + 1 < mlp_stat_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        out.close();
        std::cout << "\n[✓] Statistical benchmark results successfully exported to: " << json_output << "\n";
    }

    return 0;
}
