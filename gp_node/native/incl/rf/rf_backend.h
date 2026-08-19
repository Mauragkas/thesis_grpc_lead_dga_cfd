#ifndef RF_BACKEND_H
#define RF_BACKEND_H

/*
 * rf_backend.h — C-ABI for Random Forest regression backend.
 *
 * Training always runs on CPU (OpenMP-parallel, one thread per tree).
 * Inference can run on CPU, CUDA, or ROCm depending on the call used.
 *
 * The opaque RfModelHandle owns the forest; free it with rf_model_free().
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                         */
/* ------------------------------------------------------------------ */
typedef enum {
    RF_SUCCESS                  =  0,
    RF_ERROR_NULL_POINTER       = -1,
    RF_ERROR_DIMENSION_MISMATCH = -2,
    RF_ERROR_INVALID_PARAM      = -3,
    RF_ERROR_CUDA_FAIL          = -4,
    RF_ERROR_ROCM_FAIL          = -5,
    RF_ERROR_ALLOC_FAIL         = -6,
    RF_ERROR_UNKNOWN            = -99
} RfStatusCode;

/* ------------------------------------------------------------------ */
/* Hyperparameters                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t n_estimators;    /* Number of trees                        */
    uint32_t max_depth;       /* Maximum tree depth (0 = unlimited)     */
    uint32_t max_features;    /* Candidate features per split           */
    uint32_t min_samples_leaf;/* Minimum samples at a leaf              */
    uint64_t seed;            /* Global RNG seed; per-tree = seed+tree  */
} RfHyperparams;

/* ------------------------------------------------------------------ */
/* Opaque model handle                                                   */
/* ------------------------------------------------------------------ */
typedef struct RfModelHandle RfModelHandle;

/* ------------------------------------------------------------------ */
/* Training (always CPU + OpenMP)                                        */
/* ------------------------------------------------------------------ */

/**
 * Trains a Random Forest regressor.
 *
 * @param X_train       Row-major (n_train × dim) feature matrix
 * @param y_train       Target vector (n_train)
 * @param n_train       Number of training samples
 * @param dim           Feature dimension
 * @param params        Hyperparameters
 * @param out_handle    Output forest handle
 */
RfStatusCode rf_model_train(
    const double*        X_train,
    const double*        y_train,
    size_t               n_train,
    size_t               dim,
    const RfHyperparams* params,
    RfModelHandle**      out_handle
);

/**
 * Frees all memory associated with an RfModelHandle.
 */
void rf_model_free(RfModelHandle* handle);

/* ------------------------------------------------------------------ */
/* Inference — CPU (OpenMP)                                              */
/* ------------------------------------------------------------------ */

/**
 * Predicts targets for n_test query points; out_pred must be
 * caller-allocated (n_test doubles).
 * If out_tree_preds is non-NULL it must be (n_estimators × n_test)
 * and receives the per-tree raw predictions (for variance estimation).
 */
RfStatusCode rf_predict(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds  /* may be NULL */
);

/* ------------------------------------------------------------------ */
/* Inference — CUDA (parallel per tree × test point)                    */
/* ------------------------------------------------------------------ */
RfStatusCode rf_predict_cuda(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
);

/* ------------------------------------------------------------------ */
/* Inference — ROCm                                                      */
/* ------------------------------------------------------------------ */
RfStatusCode rf_predict_rocm(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
);

/* ------------------------------------------------------------------ */
/* Metadata                                                              */
/* ------------------------------------------------------------------ */
uint32_t rf_get_n_estimators(const RfModelHandle* handle);
uint32_t rf_get_max_depth(const RfModelHandle* handle);
size_t   rf_get_n_nodes_total(const RfModelHandle* handle);

#ifdef __cplusplus
}
#endif

#endif /* RF_BACKEND_H */
