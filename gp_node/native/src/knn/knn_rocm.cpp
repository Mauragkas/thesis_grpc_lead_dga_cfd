/*
 * knn_rocm.cpp — ROCm/HIP stub for the k-NN regression backend.
 *
 * ROCm/HIP implementation reserved; falls back to CPU until HIP runtime is linked.
 *
 * When the HIP runtime and hipcc toolchain are available, replace the body of
 * knn_predict_rocm with a HIP kernel following the same pattern as knn_cuda.cu.
 */

#include "knn_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare the CPU implementation so we can delegate to it. */
KnnStatusCode knn_predict(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ------------------------------------------------------------------ */
/* knn_predict_rocm                                                     */
/* ------------------------------------------------------------------ */

extern "C"
KnnStatusCode knn_predict_rocm(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
) {
    /* ROCm/HIP implementation reserved; falls back to CPU until HIP runtime is linked. */
    return knn_predict(handle, X_test, n_test, out_pred, out_dist);
}
