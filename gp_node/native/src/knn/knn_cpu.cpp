/*
 * knn_cpu.cpp — CPU (OpenMP) implementation of the k-NN regression backend.
 *
 * Implements:
 *   knn_model_create   — allocate + fill KnnModelHandle
 *   knn_model_free     — deallocate KnnModelHandle
 *   knn_predict        — OpenMP-parallel inverse-distance weighted k-NN
 *   knn_predict_cuda   — CPU fallback (real CUDA kernel is in knn_cuda.cu)
 *   knn_predict_rocm   — CPU fallback (real ROCm kernel is in knn_rocm.cpp)
 */

#include "knn_backend.h"
#include "knn_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* File-local helpers                                                   */
/* ------------------------------------------------------------------ */

namespace {

/**
 * Scaled squared Euclidean distance between two dim-vectors.
 *   d2 = sum_d ((a[d] - b[d]) / ls[d])^2
 */
inline double scaled_dist_sq(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ ls,
    size_t dim
) noexcept {
    double sum = 0.0;
    for (size_t d = 0; d < dim; ++d) {
        const double diff = (a[d] - b[d]) / ls[d];
        sum += diff * diff;
    }
    return sum;
}

} // anonymous namespace

/* ------------------------------------------------------------------ */
/* knn_model_create                                                     */
/* ------------------------------------------------------------------ */

extern "C"
KnnStatusCode knn_model_create(
    const double*    X_train,
    const double*    y_train,
    size_t           n_train,
    size_t           dim,
    const double*    length_scales,
    uint32_t         k,
    KnnModelHandle** out_handle
) {
    /* --- validation ------------------------------------------------- */
    if (!X_train || !y_train || !length_scales || !out_handle) {
        return KNN_ERROR_NULL_POINTER;
    }
    if (n_train == 0 || dim == 0) {
        return KNN_ERROR_DIMENSION_MISMATCH;
    }
    if (k == 0 || static_cast<size_t>(k) > n_train) {
        return KNN_ERROR_INVALID_K;
    }

    /* --- allocate --------------------------------------------------- */
    KnnModelHandle* h = nullptr;
    try {
        h = new KnnModelHandle();
        h->n_train       = n_train;
        h->dim           = dim;
        h->k             = k;
        h->X_train.assign(X_train, X_train + n_train * dim);
        h->y_train.assign(y_train, y_train + n_train);
        h->length_scales.assign(length_scales, length_scales + dim);
    } catch (...) {
        delete h;
        return KNN_ERROR_ALLOC_FAIL;
    }

    *out_handle = h;
    return KNN_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* knn_model_free                                                       */
/* ------------------------------------------------------------------ */

extern "C"
void knn_model_free(KnnModelHandle* handle) {
    delete handle; /* safe on nullptr */
}

/* ------------------------------------------------------------------ */
/* knn_predict — OpenMP parallel inverse-distance weighted k-NN        */
/* ------------------------------------------------------------------ */

extern "C"
KnnStatusCode knn_predict(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
) {
    /* --- validation ------------------------------------------------- */
    if (!handle || !X_test || !out_pred) {
        return KNN_ERROR_NULL_POINTER;
    }
    if (n_test == 0) {
        return KNN_ERROR_DIMENSION_MISMATCH;
    }

    const size_t   n_train = handle->n_train;
    const size_t   dim     = handle->dim;
    const uint32_t k       = handle->k;

    const double* Xtr = handle->X_train.data();
    const double* ytr = handle->y_train.data();
    const double* ls  = handle->length_scales.data();

    /* --- parallel loop over test points ----------------------------- */
    #pragma omp parallel for schedule(dynamic, 8)
    for (size_t m = 0; m < n_test; ++m) {
        const double* x_test_m = &X_test[m * dim];

        /* Build (dist², index) pairs for every training point. */
        std::vector<std::pair<double, size_t>> dists(n_train);
        for (size_t i = 0; i < n_train; ++i) {
            dists[i] = { scaled_dist_sq(x_test_m, &Xtr[i * dim], ls, dim), i };
        }

        /* Partial-sort: bring the k smallest distances to the front. */
        std::partial_sort(
            dists.begin(),
            dists.begin() + static_cast<ptrdiff_t>(k),
            dists.end(),
            [](const std::pair<double, size_t>& a,
               const std::pair<double, size_t>& b) {
                return a.first < b.first;
            }
        );

        /* Inverse-distance weighted prediction.
         *   w_i  = 1 / (sqrt(d2_i) + eps)   (eps avoids division by zero)
         *   pred = sum(w_i * y_i) / sum(w_i)
         */
        constexpr double kEps = 1e-12;
        double w_sum  = 0.0;
        double wy_sum = 0.0;
        double d_sum  = 0.0;

        for (uint32_t ki = 0; ki < k; ++ki) {
            const double d2  = dists[ki].first;
            const size_t idx = dists[ki].second;
            const double d   = std::sqrt(d2);
            const double w   = 1.0 / (d + kEps);
            w_sum  += w;
            wy_sum += w * ytr[idx];
            d_sum  += d;
        }

        out_pred[m] = wy_sum / w_sum;

        if (out_dist) {
            /* Mean Euclidean distance (in scaled space) to the k neighbours. */
            out_dist[m] = d_sum / static_cast<double>(k);
        }
    }

    return KNN_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* knn_predict_cuda — CPU fallback                                      */
/* Compiled only when the real CUDA TU (knn_cuda.cu) is NOT linked.    */
/* ------------------------------------------------------------------ */

#ifndef ENABLE_CUDA
extern "C"
KnnStatusCode knn_predict_cuda(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
) {
    return knn_predict(handle, X_test, n_test, out_pred, out_dist);
}
#endif /* ENABLE_CUDA */

/* ------------------------------------------------------------------ */
/* knn_predict_rocm — CPU fallback                                      */
/* (Real ROCm implementation is in knn_rocm.cpp)                       */
/* ------------------------------------------------------------------ */

extern "C"
KnnStatusCode knn_predict_rocm(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
) {
    return knn_predict(handle, X_test, n_test, out_pred, out_dist);
}
