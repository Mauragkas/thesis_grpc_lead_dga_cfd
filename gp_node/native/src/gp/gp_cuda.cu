#include "gp_cuda.h"
#include "gp_cpu.h"
#include <cmath>
#include <vector>

#if defined(__CUDACC__) || defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

__device__ inline double d_compute_dist_sq(
    const double* __restrict__ x1,
    const double* __restrict__ x2,
    size_t dim,
    const double* __restrict__ ls
) {
    double sum = 0.0;
    for (size_t d = 0; d < dim; ++d) {
        double diff = (x1[d] - x2[d]) / ls[d];
        sum += diff * diff;
    }
    return sum;
}

__device__ inline double d_eval_kernel(
    double d2,
    GpKernelType ktype,
    double sig_var
) {
    if (ktype == GP_KERNEL_MATERN52) {
        double dist = sqrt(d2 > 0.0 ? d2 : 0.0);
        double sqrt5_r = 2.23606797749979 * dist;
        return sig_var * (1.0 + sqrt5_r + (5.0 / 3.0) * d2) * exp(-sqrt5_r);
    } else {
        return sig_var * exp(-0.5 * d2);
    }
}

__global__ void k_compute_covariance(
    const double* __restrict__ X1,
    size_t n1,
    const double* __restrict__ X2,
    size_t n2,
    size_t dim,
    const double* __restrict__ ls,
    double sig_var,
    double noise_var,
    GpKernelType ktype,
    bool add_noise,
    bool is_symmetric,
    double* __restrict__ out_K
) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n1 && j < n2) {
        if (is_symmetric && j < i) {
            return;
        }
        const double* x1 = &X1[i * dim];
        const double* x2 = &X2[j * dim];
        double d2 = d_compute_dist_sq(x1, x2, dim, ls);
        double val = d_eval_kernel(d2, ktype, sig_var);
        if (add_noise && i == j) {
            val += noise_var;
        }
        out_K[i * n2 + j] = val;
        if (is_symmetric && i != j) {
            out_K[j * n1 + i] = val;
        }
    }
}

__global__ void k_predict_mean(
    const double* __restrict__ X_train,
    size_t n_train,
    const double* __restrict__ X_test,
    size_t n_test,
    size_t dim,
    const double* __restrict__ alpha,
    const double* __restrict__ ls,
    double sig_var,
    GpKernelType ktype,
    double* __restrict__ out_mean
) {
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < n_test) {
        const double* x_test_m = &X_test[m * dim];
        double mean_val = 0.0;
        for (size_t i = 0; i < n_train; ++i) {
            const double* x_train_i = &X_train[i * dim];
            double d2 = d_compute_dist_sq(x_test_m, x_train_i, dim, ls);
            double k_val = d_eval_kernel(d2, ktype, sig_var);
            mean_val += k_val * alpha[i];
        }
        out_mean[m] = mean_val;
    }
}

} // namespace

bool cuda_is_available(int* out_device_count) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        if (out_device_count) *out_device_count = 0;
        return false;
    }
    if (out_device_count) *out_device_count = count;
    return true;
}

GpStatusCode cuda_compute_covariance(
    int device_id,
    const double* X1,
    size_t n1,
    const double* X2,
    size_t n2,
    size_t dim,
    const GpHyperparams* params,
    bool add_diagonal_noise,
    double* out_K
) {
    if (!X1 || !X2 || !params || !out_K || !params->length_scales) return GP_ERROR_NULL_POINTER;
    if (dim == 0 || n1 == 0 || n2 == 0) return GP_ERROR_DIMENSION_MISMATCH;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return GP_ERROR_CUDA_FAIL;

    double *d_X1 = nullptr, *d_X2 = nullptr, *d_ls = nullptr, *d_K = nullptr;
    size_t bytes_X1 = n1 * dim * sizeof(double);
    size_t bytes_X2 = n2 * dim * sizeof(double);
    size_t bytes_ls = dim * sizeof(double);
    size_t bytes_K = n1 * n2 * sizeof(double);

    bool is_symmetric = (X1 == X2 && n1 == n2);

    if (cudaMalloc(&d_X1, bytes_X1) != cudaSuccess ||
        (is_symmetric ? (d_X2 = d_X1, false) : (cudaMalloc(&d_X2, bytes_X2) != cudaSuccess)) ||
        cudaMalloc(&d_ls, bytes_ls) != cudaSuccess ||
        cudaMalloc(&d_K, bytes_K) != cudaSuccess) {
        if (d_X1) cudaFree(d_X1);
        if (d_X2 && d_X2 != d_X1) cudaFree(d_X2);
        if (d_ls) cudaFree(d_ls);
        if (d_K) cudaFree(d_K);
        return GP_ERROR_CUDA_FAIL;
    }

    cudaMemcpy(d_X1, X1, bytes_X1, cudaMemcpyHostToDevice);
    if (!is_symmetric) {
        cudaMemcpy(d_X2, X2, bytes_X2, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_ls, params->length_scales, bytes_ls, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n2 + block.x - 1) / block.x, (n1 + block.y - 1) / block.y);

    k_compute_covariance<<<grid, block>>>(
        d_X1, n1, d_X2, n2, dim, d_ls,
        params->signal_variance, params->noise_variance,
        params->kernel_type, add_diagonal_noise, is_symmetric, d_K
    );

    cudaDeviceSynchronize();
    cudaMemcpy(out_K, d_K, bytes_K, cudaMemcpyDeviceToHost);

    cudaFree(d_X1);
    if (!is_symmetric) cudaFree(d_X2);
    cudaFree(d_ls);
    cudaFree(d_K);

    return GP_SUCCESS;
}

GpStatusCode cuda_cholesky(
    int device_id,
    const double* K,
    size_t n,
    double* out_L
) {
    (void)device_id;
    return cpu_cholesky(K, n, out_L);
}

GpStatusCode cuda_predict_batch(
    int device_id,
    const double* X_train,
    size_t n_train,
    const double* X_test,
    size_t n_test,
    size_t dim,
    const double* alpha,
    const double* L,
    const GpHyperparams* params,
    double* out_mean,
    double* out_var
) {
    if (!X_train || !X_test || !alpha || !params || !out_mean) return GP_ERROR_NULL_POINTER;
    if (n_train == 0 || n_test == 0 || dim == 0) return GP_ERROR_DIMENSION_MISMATCH;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return GP_ERROR_CUDA_FAIL;

    double *d_Xtr = nullptr, *d_Xte = nullptr, *d_alpha = nullptr, *d_ls = nullptr, *d_mean = nullptr;
    size_t bytes_Xtr = n_train * dim * sizeof(double);
    size_t bytes_Xte = n_test * dim * sizeof(double);
    size_t bytes_alpha = n_train * sizeof(double);
    size_t bytes_ls = dim * sizeof(double);
    size_t bytes_mean = n_test * sizeof(double);

    if (cudaMalloc(&d_Xtr, bytes_Xtr) != cudaSuccess ||
        cudaMalloc(&d_Xte, bytes_Xte) != cudaSuccess ||
        cudaMalloc(&d_alpha, bytes_alpha) != cudaSuccess ||
        cudaMalloc(&d_ls, bytes_ls) != cudaSuccess ||
        cudaMalloc(&d_mean, bytes_mean) != cudaSuccess) {
        if (d_Xtr) cudaFree(d_Xtr);
        if (d_Xte) cudaFree(d_Xte);
        if (d_alpha) cudaFree(d_alpha);
        if (d_ls) cudaFree(d_ls);
        if (d_mean) cudaFree(d_mean);
        return GP_ERROR_CUDA_FAIL;
    }

    cudaMemcpy(d_Xtr, X_train, bytes_Xtr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xte, X_test, bytes_Xte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha, bytes_alpha, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls, params->length_scales, bytes_ls, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n_test + block - 1) / block;

    k_predict_mean<<<grid, block>>>(
        d_Xtr, n_train, d_Xte, n_test, dim, d_alpha, d_ls,
        params->signal_variance, params->kernel_type, d_mean
    );

    cudaDeviceSynchronize();
    cudaMemcpy(out_mean, d_mean, bytes_mean, cudaMemcpyDeviceToHost);

    cudaFree(d_Xtr);
    cudaFree(d_Xte);
    cudaFree(d_alpha);
    cudaFree(d_ls);
    cudaFree(d_mean);

    if (out_var && L) {
        double* temp_mean = new double[n_test];
        cpu_predict_batch(X_train, n_train, X_test, n_test, dim, alpha, L, params, temp_mean, out_var);
        delete[] temp_mean;
    }

    return GP_SUCCESS;
}

#else

bool cuda_is_available(int* out_device_count) {
    if (out_device_count) *out_device_count = 0;
    return false;
}

GpStatusCode cuda_compute_covariance(
    int, const double*, size_t, const double*, size_t, size_t, const GpHyperparams*, bool, double*
) {
    return GP_ERROR_CUDA_FAIL;
}

GpStatusCode cuda_cholesky(int, const double*, size_t, double*) {
    return GP_ERROR_CUDA_FAIL;
}

GpStatusCode cuda_predict_batch(
    int, const double*, size_t, const double*, size_t, size_t, const double*, const double*, const GpHyperparams*, double*, double*
) {
    return GP_ERROR_CUDA_FAIL;
}

#endif
