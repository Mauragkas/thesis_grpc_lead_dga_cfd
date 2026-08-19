#ifndef MLP_INTERNAL_H
#define MLP_INTERNAL_H

#include "mlp_backend.h"
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>

struct MlpLayer {
    size_t in_features;
    size_t out_features;
    std::vector<double> weights; // Row-major: [in_features * out_features], w_{i, j} is from in_i to out_j
    std::vector<double> biases;  // [out_features]
};

struct MlpModelHandle {
    size_t input_dim;
    std::vector<MlpLayer> layers;
    MlpActivation activation;

    size_t total_parameters() const {
        size_t total = 0;
        for (const auto& layer : layers) {
            total += layer.weights.size() + layer.biases.size();
        }
        return total;
    }
};

// Activation and derivative helpers
inline double mlp_activate(double x, MlpActivation act) {
    switch (act) {
        case MLP_ACT_RELU:
            return x > 0.0 ? x : 0.0;
        case MLP_ACT_SILU:
            return x / (1.0 + std::exp(-x));
        case MLP_ACT_GELU: {
            // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            double x3 = x * x * x;
            return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x3)));
        }
        case MLP_ACT_TANH:
            return std::tanh(x);
    }
    return x;
}

inline double mlp_activate_deriv(double x, double act_val, MlpActivation act) {
    switch (act) {
        case MLP_ACT_RELU:
            return x > 0.0 ? 1.0 : 0.0;
        case MLP_ACT_SILU: {
            double sig = 1.0 / (1.0 + std::exp(-x));
            return sig * (1.0 + x * (1.0 - sig));
        }
        case MLP_ACT_GELU: {
            // Derivative of 0.5 * x * (1 + erf(x / sqrt(2)))
            double cdf = 0.5 * (1.0 + std::erf(x * 0.7071067811865475));
            double pdf = 0.3989422804014327 * std::exp(-0.5 * x * x);
            return cdf + x * pdf;
        }
        case MLP_ACT_TANH:
            return 1.0 - act_val * act_val;
    }
    return 1.0;
}

#endif /* MLP_INTERNAL_H */
