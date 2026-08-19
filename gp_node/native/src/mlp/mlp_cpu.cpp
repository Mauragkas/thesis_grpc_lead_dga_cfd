#include "mlp_backend.h"
#include "mlp_internal.h"
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass for a single sample
// ─────────────────────────────────────────────────────────────────────────────
static void mlp_forward_sample(
    const MlpModelHandle* model,
    const double* x_in,
    std::vector<std::vector<double>>& layer_inputs, // Pre-activations z
    std::vector<std::vector<double>>& layer_outputs // Post-activations a
) {
    size_t num_layers = model->layers.size();
    layer_inputs.resize(num_layers);
    layer_outputs.resize(num_layers);

    const double* curr_input = x_in;
    for (size_t l = 0; l < num_layers; ++l) {
        const auto& layer = model->layers[l];
        layer_inputs[l].resize(layer.out_features);
        layer_outputs[l].resize(layer.out_features);

        bool is_last = (l == num_layers - 1);

        for (size_t j = 0; j < layer.out_features; ++j) {
            double sum = layer.biases[j];
            for (size_t i = 0; i < layer.in_features; ++i) {
                sum += curr_input[i] * layer.weights[i * layer.out_features + j];
            }
            layer_inputs[l][j] = sum;
            layer_outputs[l][j] = is_last ? sum : mlp_activate(sum, model->activation);
        }
        curr_input = layer_outputs[l].data();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Training with Adam Optimizer
// ─────────────────────────────────────────────────────────────────────────────
extern "C" MlpStatusCode mlp_model_train(
    const double*         X_train,
    const double*         y_train,
    size_t                n_train,
    size_t                dim,
    const MlpHyperparams* params,
    MlpModelHandle**      out_handle
) {
    if (!X_train || !y_train || !params || !out_handle) return MLP_ERROR_NULL_POINTER;
    if (n_train == 0 || dim == 0) return MLP_ERROR_DIMENSION_MISMATCH;
    if (params->num_hidden_layers == 0 || params->num_hidden_layers > 4) return MLP_ERROR_INVALID_PARAM;

    auto* model = new (std::nothrow) MlpModelHandle();
    if (!model) return MLP_ERROR_ALLOC_FAIL;

    model->input_dim = dim;
    model->activation = params->activation;

    // Build layer topology: [dim, hidden[0], hidden[1], ..., 1]
    std::vector<size_t> topology;
    topology.push_back(dim);
    for (size_t i = 0; i < params->num_hidden_layers; ++i) {
        if (params->hidden_layers[i] == 0) break;
        topology.push_back(params->hidden_layers[i]);
    }
    topology.push_back(1); // Single regression output

    size_t num_layers = topology.size() - 1;
    model->layers.resize(num_layers);

    std::mt19937_64 rng(params->seed == 0 ? 42 : params->seed);

    // Initialize weights with He / Xavier uniform initialization
    for (size_t l = 0; l < num_layers; ++l) {
        size_t in_f = topology[l];
        size_t out_f = topology[l + 1];

        model->layers[l].in_features = in_f;
        model->layers[l].out_features = out_f;
        model->layers[l].weights.resize(in_f * out_f);
        model->layers[l].biases.assign(out_f, 0.0);

        double limit = std::sqrt(6.0 / (in_f + out_f));
        std::uniform_real_distribution<double> dist(-limit, limit);

        for (auto& w : model->layers[l].weights) {
            w = dist(rng);
        }
    }

    // Adam optimizer moments
    std::vector<std::vector<double>> m_w(num_layers), v_w(num_layers);
    std::vector<std::vector<double>> m_b(num_layers), v_b(num_layers);

    for (size_t l = 0; l < num_layers; ++l) {
        m_w[l].assign(model->layers[l].weights.size(), 0.0);
        v_w[l].assign(model->layers[l].weights.size(), 0.0);
        m_b[l].assign(model->layers[l].biases.size(), 0.0);
        v_b[l].assign(model->layers[l].biases.size(), 0.0);
    }

    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    const double lr = params->learning_rate > 0.0 ? params->learning_rate : 1e-3;
    const double decay = params->weight_decay >= 0.0 ? params->weight_decay : 1e-4;
    const size_t batch_size = params->batch_size > 0 ? std::min((size_t)params->batch_size, n_train) : std::min((size_t)32, n_train);
    const size_t epochs = params->epochs > 0 ? params->epochs : 300;

    std::vector<size_t> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);

    size_t timestep = 0;

    // Pre-allocated per-thread scratch buffers for forward and backward passes
    std::vector<std::vector<double>> z_buf, a_buf;
    std::vector<std::vector<double>> delta(num_layers);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t b_start = 0; b_start < n_train; b_start += batch_size) {
            size_t b_end = std::min(b_start + batch_size, n_train);
            size_t curr_batch = b_end - b_start;
            if (curr_batch == 0) continue;

            timestep++;

            // Gradients accumulators for this batch
            std::vector<std::vector<double>> grad_w(num_layers);
            std::vector<std::vector<double>> grad_b(num_layers);
            for (size_t l = 0; l < num_layers; ++l) {
                grad_w[l].assign(model->layers[l].weights.size(), 0.0);
                grad_b[l].assign(model->layers[l].biases.size(), 0.0);
            }

            // Accumulate gradients across batch
            for (size_t idx = b_start; idx < b_end; ++idx) {
                size_t sample_idx = indices[idx];
                const double* x_sample = &X_train[sample_idx * dim];
                double y_sample = y_train[sample_idx];

                // Forward pass
                mlp_forward_sample(model, x_sample, z_buf, a_buf);

                // Backward pass: Output layer error (MSE derivative: pred - target)
                double pred = a_buf.back()[0];
                double out_err = (pred - y_sample);

                delta.back().resize(1);
                delta.back()[0] = out_err;

                // Backpropagate error through hidden layers
                for (int l = (int)num_layers - 2; l >= 0; --l) {
                    const auto& next_layer = model->layers[l + 1];
                    const auto& curr_layer = model->layers[l];
                    delta[l].resize(curr_layer.out_features);

                    for (size_t j = 0; j < curr_layer.out_features; ++j) {
                        double sum = 0.0;
                        for (size_t k = 0; k < next_layer.out_features; ++k) {
                            sum += delta[l + 1][k] * next_layer.weights[j * next_layer.out_features + k];
                        }
                        double deriv = mlp_activate_deriv(z_buf[l][j], a_buf[l][j], model->activation);
                        delta[l][j] = sum * deriv;
                    }
                }

                // Accumulate gradients
                for (size_t l = 0; l < num_layers; ++l) {
                    const double* prev_act = (l == 0) ? x_sample : a_buf[l - 1].data();
                    size_t in_f = model->layers[l].in_features;
                    size_t out_f = model->layers[l].out_features;

                    for (size_t i = 0; i < in_f; ++i) {
                        for (size_t j = 0; j < out_f; ++j) {
                            grad_w[l][i * out_f + j] += delta[l][j] * prev_act[i];
                        }
                    }
                    for (size_t j = 0; j < out_f; ++j) {
                        grad_b[l][j] += delta[l][j];
                    }
                }
            }

            // Adam update step with weight decay
            double beta1_t = std::pow(beta1, (double)timestep);
            double beta2_t = std::pow(beta2, (double)timestep);
            double inv_batch = 1.0 / (double)curr_batch;

            for (size_t l = 0; l < num_layers; ++l) {
                auto& layer = model->layers[l];
                size_t num_w = layer.weights.size();
                size_t num_b = layer.biases.size();

                for (size_t i = 0; i < num_w; ++i) {
                    double g = grad_w[l][i] * inv_batch + decay * layer.weights[i];
                    m_w[l][i] = beta1 * m_w[l][i] + (1.0 - beta1) * g;
                    v_w[l][i] = beta2 * v_w[l][i] + (1.0 - beta2) * g * g;

                    double m_hat = m_w[l][i] / (1.0 - beta1_t);
                    double v_hat = v_w[l][i] / (1.0 - beta2_t);

                    layer.weights[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                }

                for (size_t j = 0; j < num_b; ++j) {
                    double g = grad_b[l][j] * inv_batch;
                    m_b[l][j] = beta1 * m_b[l][j] + (1.0 - beta1) * g;
                    v_b[l][j] = beta2 * v_b[l][j] + (1.0 - beta2) * g * g;

                    double m_hat = m_b[l][j] / (1.0 - beta1_t);
                    double v_hat = v_b[l][j] / (1.0 - beta2_t);

                    layer.biases[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                }
            }
        }
    }

    *out_handle = model;
    return MLP_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction / Inference (OpenMP parallel across test queries)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" MlpStatusCode mlp_predict(
    const MlpModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred
) {
    if (!handle || !X_test || !out_pred) return MLP_ERROR_NULL_POINTER;
    if (n_test == 0) return MLP_SUCCESS;

    size_t dim = handle->input_dim;
    size_t num_layers = handle->layers.size();

    #pragma omp parallel
    {
        std::vector<std::vector<double>> local_z(num_layers);
        std::vector<std::vector<double>> local_a(num_layers);

        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_test; ++t) {
            const double* x_sample = &X_test[t * dim];
            mlp_forward_sample(handle, x_sample, local_z, local_a);
            out_pred[t] = local_a.back()[0];
        }
    }

    return MLP_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cleanup & Metadata
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void mlp_model_free(MlpModelHandle* handle) {
    delete handle;
}

extern "C" uint32_t mlp_get_num_layers(const MlpModelHandle* handle) {
    return handle ? (uint32_t)handle->layers.size() : 0;
}

extern "C" size_t mlp_get_total_parameters(const MlpModelHandle* handle) {
    return handle ? handle->total_parameters() : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU fallback stubs for CUDA / ROCm when compiled without GPU acceleration
// ─────────────────────────────────────────────────────────────────────────────
#ifndef ENABLE_CUDA
extern "C" MlpStatusCode mlp_predict_cuda(
    const MlpModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred
) {
    return mlp_predict(handle, X_test, n_test, out_pred);
}
#endif

extern "C" MlpStatusCode mlp_predict_rocm(
    const MlpModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred
) {
    return mlp_predict(handle, X_test, n_test, out_pred);
}
