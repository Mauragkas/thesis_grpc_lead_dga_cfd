#ifndef MLP_BACKEND_H
#define MLP_BACKEND_H

/*
 * mlp_backend.h — C-ABI for Multi-Layer Perceptron (Neural Network) regression backend.
 *
 * Training runs on CPU (OpenMP-accelerated SGD/Adam).
 * Inference can run on CPU, CUDA, or ROCm.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MLP_SUCCESS                  =  0,
    MLP_ERROR_NULL_POINTER       = -1,
    MLP_ERROR_DIMENSION_MISMATCH = -2,
    MLP_ERROR_INVALID_PARAM      = -3,
    MLP_ERROR_CUDA_FAIL          = -4,
    MLP_ERROR_ROCM_FAIL          = -5,
    MLP_ERROR_ALLOC_FAIL         = -6,
    MLP_ERROR_UNKNOWN            = -99
} MlpStatusCode;

typedef enum {
    MLP_ACT_RELU = 0,
    MLP_ACT_SILU = 1,
    MLP_ACT_GELU = 2,
    MLP_ACT_TANH = 3
} MlpActivation;

typedef struct {
    uint32_t hidden_layers[4];   /* Up to 4 hidden layer sizes, e.g. [64, 32, 0, 0] */
    uint32_t num_hidden_layers;  /* Number of hidden layers (1 to 4) */
    double   learning_rate;      /* Adam learning rate (e.g. 0.001) */
    double   weight_decay;       /* L2 regularization factor (e.g. 1e-4) */
    uint32_t batch_size;         /* Mini-batch size (e.g. 32) */
    uint32_t epochs;             /* Number of training epochs (e.g. 300) */
    MlpActivation activation;    /* Activation function */
    uint64_t seed;               /* Random initialization seed */
} MlpHyperparams;

typedef struct MlpModelHandle MlpModelHandle;

/**
 * Trains a Multi-Layer Perceptron regressor on scaled feature matrix X_train and targets y_train.
 */
MlpStatusCode mlp_model_train(
    const double*         X_train,
    const double*         y_train,
    size_t                n_train,
    size_t                dim,
    const MlpHyperparams* params,
    MlpModelHandle**      out_handle
);

/**
 * Frees all memory associated with an MlpModelHandle.
 */
void mlp_model_free(MlpModelHandle* handle);

/**
 * Predicts targets for n_test query points using CPU OpenMP SIMD.
 */
MlpStatusCode mlp_predict(
    const MlpModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred
);

/**
 * Predicts targets for n_test query points using NVIDIA CUDA GPU.
 */
MlpStatusCode mlp_predict_cuda(
    const MlpModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred
);

/**
 * Predicts targets for n_test query points using AMD ROCm GPU.
 */
MlpStatusCode mlp_predict_rocm(
    const MlpModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred
);

uint32_t mlp_get_num_layers(const MlpModelHandle* handle);
size_t   mlp_get_total_parameters(const MlpModelHandle* handle);

#ifdef __cplusplus
}
#endif

#endif /* MLP_BACKEND_H */
