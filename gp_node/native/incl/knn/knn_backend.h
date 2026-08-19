#ifndef KNN_BACKEND_H
#define KNN_BACKEND_H

/*
 * knn_backend.h — C-ABI for k-Nearest Neighbours regression backend.
 *
 * All functions return KnnStatusCode; on success the output pointers
 * are populated. No heap memory is owned by the caller except the
 * KnnModelHandle returned by knn_model_create(), which must be freed
 * with knn_model_free().
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                         */
/* ------------------------------------------------------------------ */
typedef enum {
    KNN_SUCCESS                  =  0,
    KNN_ERROR_NULL_POINTER       = -1,
    KNN_ERROR_DIMENSION_MISMATCH = -2,
    KNN_ERROR_INVALID_K          = -3,
    KNN_ERROR_CUDA_FAIL          = -4,
    KNN_ERROR_ROCM_FAIL          = -5,
    KNN_ERROR_ALLOC_FAIL         = -6,
    KNN_ERROR_UNKNOWN            = -99
} KnnStatusCode;

/* ------------------------------------------------------------------ */
/* Opaque model handle (owns training data + feature scales)           */
/* ------------------------------------------------------------------ */
typedef struct KnnModelHandle KnnModelHandle;

/* ------------------------------------------------------------------ */
/* Training                                                             */
/* ------------------------------------------------------------------ */

/**
 * Fits a k-NN model (copies X_train, y_train, and length_scales into
 * the handle).  length_scales[d] is the per-feature standard deviation
 * used to normalise Euclidean distances.
 *
 * @param X_train       Row-major (n_train × dim) feature matrix
 * @param y_train       Target vector (n_train)
 * @param n_train       Number of training samples
 * @param dim           Feature dimension
 * @param length_scales Per-feature scale factors (dim)
 * @param k             Number of nearest neighbours
 * @param out_handle    Output handle; caller must free with knn_model_free()
 */
KnnStatusCode knn_model_create(
    const double* X_train,
    const double* y_train,
    size_t        n_train,
    size_t        dim,
    const double* length_scales,
    uint32_t      k,
    KnnModelHandle** out_handle
);

/**
 * Frees all memory associated with a KnnModelHandle.
 */
void knn_model_free(KnnModelHandle* handle);

/* ------------------------------------------------------------------ */
/* Inference                                                            */
/* ------------------------------------------------------------------ */

/**
 * Predicts target values for n_test query points using the fitted model.
 * Uses inverse-distance weighting: w_i = 1 / (d_i + 1e-12).
 *
 * @param handle    Fitted model handle
 * @param X_test    Row-major (n_test × dim) query matrix
 * @param n_test    Number of query points
 * @param out_pred  Output predictions (n_test); caller-allocated
 * @param out_dist  If non-NULL, filled with mean k-NN distance for each
 *                  query point (useful as uncertainty proxy)
 */
KnnStatusCode knn_predict(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
);

/* ------------------------------------------------------------------ */
/* CUDA variant — same signature, implemented in knn_cuda.cu            */
/* ------------------------------------------------------------------ */
KnnStatusCode knn_predict_cuda(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
);

/* ------------------------------------------------------------------ */
/* ROCm variant — same signature, implemented in knn_rocm.cpp           */
/* ------------------------------------------------------------------ */
KnnStatusCode knn_predict_rocm(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
);

#ifdef __cplusplus
}
#endif

#endif /* KNN_BACKEND_H */
