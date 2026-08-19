/*
 * rf_rocm.cpp — ROCm/HIP inference stub for the Random Forest backend.
 *
 * ROCm/HIP inference reserved; falls back to CPU until HIP runtime is linked.
 *
 * When HIP support is eventually added, replace this file with a .hip file
 * that mirrors the structure of rf_cuda.cu: serialise trees into a flat
 * HIP device buffer, launch a hip::launch_kernel, and copy results back.
 *
 * Until then, rf_predict_rocm simply delegates to rf_predict (CPU + OpenMP).
 */

#include "rf_backend.h"

/* rf_predict is defined in rf_cpu.cpp and linked into the same library.   */
extern "C" RfStatusCode rf_predict(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
);

/* ---------------------------------------------------------------------- */
/* rf_predict_rocm                                                          */
/*                                                                          */
/* ROCm/HIP inference reserved; falls back to CPU until HIP runtime is     */
/* linked.                                                                  */
/* ---------------------------------------------------------------------- */
RfStatusCode rf_predict_rocm(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
) {
    /* ROCm/HIP inference reserved; falls back to CPU until HIP runtime is linked. */
    return rf_predict(handle, X_test, n_test, out_pred, out_tree_preds);
}
