#include "mlp_backend.h"
#include "mlp_internal.h"
#include <vector>
#include <cmath>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline double dev_activate(double x, MlpActivation act) {
    switch (act) {
        case MLP_ACT_RELU:
            return x > 0.0 ? x : 0.0;
        case MLP_ACT_SILU:
            return x / (1.0 + exp(-x));
        case MLP_ACT_GELU: {
            double x3 = x * x * x;
            return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x3)));
        }
        case MLP_ACT_TANH:
            return tanh(x);
    }
    return x;
}

__global__ void mlp_predict_kernel(
    const double* d_x_test,
    size_t n_test,
    size_t in_dim,
    const double* d_w0, const double* d_b0, size_t h0_dim,
    const double* d_w1, const double* d_b1, size_t h1_dim,
    const double* d_w_out, const double* d_b_out,
    MlpActivation act,
    double* d_out_pred
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_test) return;

    const double* x = &d_x_test[idx * in_dim];

    // Layer 0: in_dim -> h0_dim
    double h0[128];
    for (size_t j = 0; j < h0_dim; ++j) {
        double sum = d_b0[j];
        for (size_t i = 0; i < in_dim; ++i) {
            sum += x[i] * d_w0[i * h0_dim + j];
        }
        h0[j] = dev_activate(sum, act);
    }

    // Layer 1 (if present): h0_dim -> h1_dim
    double h1[128];
    const double* prev = h0;
    size_t prev_dim = h0_dim;

    if (h1_dim > 0 && d_w1 != nullptr) {
        for (size_t j = 0; j < h1_dim; ++j) {
            double sum = d_b1[j];
            for (size_t i = 0; i < h0_dim; ++i) {
                sum += h0[i] * d_w1[i * h1_dim + j];
            }
            h1[j] = dev_activate(sum, act);
        }
        prev = h1;
        prev_dim = h1_dim;
    }

    // Output layer: prev_dim -> 1
    double out_val = d_b_out[0];
    for (size_t i = 0; i < prev_dim; ++i) {
        out_val += prev[i] * d_w_out[i];
    }

    d_out_pred[idx] = out_val;
}

extern "C" MlpStatusCode mlp_predict_cuda(
    const MlpModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred
) {
    if (!handle || !X_test || !out_pred) return MLP_ERROR_NULL_POINTER;
    if (n_test == 0) return MLP_SUCCESS;

    size_t num_layers = handle->layers.size();
    if (num_layers < 2 || num_layers > 3) {
        // Fallback to CPU for general topologies
        return mlp_predict(handle, X_test, n_test, out_pred);
    }

    size_t dim = handle->input_dim;
    size_t h0_dim = handle->layers[0].out_features;
    size_t h1_dim = (num_layers == 3) ? handle->layers[1].out_features : 0;

    double *d_x = nullptr, *d_out = nullptr;
    double *d_w0 = nullptr, *d_b0 = nullptr;
    double *d_w1 = nullptr, *d_b1 = nullptr;
    double *d_w_out = nullptr, *d_b_out = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_x, n_test * dim * sizeof(double));
    if (err != cudaSuccess) return MLP_ERROR_CUDA_FAIL;
    err = cudaMalloc(&d_out, n_test * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_x); return MLP_ERROR_CUDA_FAIL; }

    cudaMemcpy(d_x, X_test, n_test * dim * sizeof(double), cudaMemcpyHostToDevice);

    // Copy Layer 0
    cudaMalloc(&d_w0, handle->layers[0].weights.size() * sizeof(double));
    cudaMalloc(&d_b0, handle->layers[0].biases.size() * sizeof(double));
    cudaMemcpy(d_w0, handle->layers[0].weights.data(), handle->layers[0].weights.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b0, handle->layers[0].biases.data(), handle->layers[0].biases.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Copy Layer 1 (if 3 layers)
    if (num_layers == 3) {
        cudaMalloc(&d_w1, handle->layers[1].weights.size() * sizeof(double));
        cudaMalloc(&d_b1, handle->layers[1].biases.size() * sizeof(double));
        cudaMemcpy(d_w1, handle->layers[1].weights.data(), handle->layers[1].weights.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, handle->layers[1].biases.data(), handle->layers[1].biases.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Copy Output Layer
    const auto& out_layer = handle->layers.back();
    cudaMalloc(&d_w_out, out_layer.weights.size() * sizeof(double));
    cudaMalloc(&d_b_out, out_layer.biases.size() * sizeof(double));
    cudaMemcpy(d_w_out, out_layer.weights.data(), out_layer.weights.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_out, out_layer.biases.data(), out_layer.biases.size() * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n_test + threads - 1) / threads;

    mlp_predict_kernel<<<blocks, threads>>>(
        d_x, n_test, dim,
        d_w0, d_b0, h0_dim,
        d_w1, d_b1, h1_dim,
        d_w_out, d_b_out,
        handle->activation,
        d_out
    );

    cudaMemcpy(out_pred, d_out, n_test * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_w0);
    cudaFree(d_b0);
    if (d_w1) { cudaFree(d_w1); cudaFree(d_b1); }
    cudaFree(d_w_out);
    cudaFree(d_b_out);

    return MLP_SUCCESS;
}
#endif
