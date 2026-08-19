/*
 * rf_cuda.cu — CUDA-accelerated batch inference for the Random Forest backend.
 *
 * Strategy
 * --------
 *  1. All trees are serialised into two flat GPU arrays:
 *       d_nodes[]        — concatenated RfNodeGpu structs for every tree
 *       d_tree_offsets[] — d_tree_offsets[t] = start index of tree t in d_nodes
 *  2. Kernel rf_inference_kernel:
 *       grid  = (n_test, 1, 1)   — one block per test point
 *       block = (n_trees_clamped, 1, 1) — one thread per tree
 *     Each thread walks its tree and atomicAdd's into a shared accumulator.
 *     After __syncthreads(), thread 0 divides by n_trees and writes out_pred.
 *  3. On any CUDA error, falls back to rf_predict (CPU).
 *
 * Build guards
 * ------------
 *  Compiled only when ENABLE_CUDA is defined (i.e. nvcc is in use).
 *  When ENABLE_CUDA is absent the entire file is a set of CPU-fallback stubs.
 *
 * Linking notes
 * -------------
 *  When this translation unit is linked, its rf_predict_cuda definition
 *  replaces the stub in rf_cpu.cpp.  rf_predict_rocm is NOT defined here;
 *  it lives in rf_rocm.cpp.
 */

#include "rf_backend.h"
#include "rf_internal.h"

#include <cstdio>
#include <vector>

/* rf_predict is the CPU reference used as a fallback */
extern "C" RfStatusCode rf_predict(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
);

/* ======================================================================
 * CUDA path
 * ====================================================================== */
#if defined(__CUDACC__) || defined(ENABLE_CUDA)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* ---------------------------------------------------------------------- */
/* GPU-friendly node representation (all int32 / double, no pointers)      */
/* ---------------------------------------------------------------------- */
struct RfNodeGpu {
    int    feature_idx; /* -1 => leaf                                       */
    int    left;        /* child indices into the per-tree node subarray    */
    int    right;
    double threshold;
    double leaf_value;
};

/* Maximum threads-per-block for the inference kernel.
 * Capped at 1024 (hardware limit).  Trees beyond this limit are handled
 * by a reduction loop inside each thread.                                 */
static constexpr int kMaxTPB = 1024;

/* ---------------------------------------------------------------------- */
/* Helper: CUDA error check with stderr warning and fallback return code   */
/* ---------------------------------------------------------------------- */
namespace {

inline bool cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[rf_cuda] CUDA error in %s: %s\n",
                     msg, cudaGetErrorString(err));
        return false;
    }
    return true;
}

} /* anonymous namespace */

/* ---------------------------------------------------------------------- */
/* Kernel: one block per test point, one thread per tree (clamped).
 *
 * Each thread walks its assigned tree in d_nodes (offset by d_tree_offsets)
 * and atomicAdd's its prediction into shared memory.  Thread 0 writes the
 * averaged result to out_pred and, if requested, fills out_tree_preds.   */
/* ---------------------------------------------------------------------- */
__global__ static void rf_inference_kernel(
    const RfNodeGpu* __restrict__ d_nodes,
    const int*       __restrict__ d_tree_offsets,
    int                           n_trees,
    const double*    __restrict__ d_X_test,
    size_t                        n_test,
    size_t                        dim,
    double*                       d_out_pred,
    double*                       d_out_tree_preds  /* may be nullptr      */
) {
    /* Each block handles one test point; blockIdx.x == test_point index   */
    const size_t test_idx = static_cast<size_t>(blockIdx.x);
    if (test_idx >= n_test) return;

    const double* x = &d_X_test[test_idx * dim];

    /* Shared accumulator: one double for the sum, initialised by thread 0 */
    extern __shared__ double s_sum[]; /* 1 element                         */
    if (threadIdx.x == 0) s_sum[0] = 0.0;
    __syncthreads();

    /*
     * Threads walk trees in a stride loop so that forests with more trees
     * than kMaxTPB are still covered.
     */
    for (int t = static_cast<int>(threadIdx.x); t < n_trees;
         t += static_cast<int>(blockDim.x)) {
        const RfNodeGpu* tree_nodes = &d_nodes[d_tree_offsets[t]];

        /* Tree traversal */
        int node = 0;
        while (tree_nodes[node].feature_idx != -1) {
            const RfNodeGpu& nd = tree_nodes[node];
            node = (x[nd.feature_idx] <= nd.threshold) ? nd.left : nd.right;
        }
        const double pred_t = tree_nodes[node].leaf_value;

        atomicAdd(&s_sum[0], pred_t);

        /* Store per-tree prediction if requested (layout: [n_trees x n_test]) */
        if (d_out_tree_preds) {
            d_out_tree_preds[static_cast<size_t>(t) * n_test + test_idx] = pred_t;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        d_out_pred[test_idx] = s_sum[0] / static_cast<double>(n_trees);
    }
}

/* ---------------------------------------------------------------------- */
/* rf_predict_cuda — public entry point                                    */
/* ---------------------------------------------------------------------- */
extern "C"
RfStatusCode rf_predict_cuda(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
) {
    if (!handle || !X_test || !out_pred) return RF_ERROR_NULL_POINTER;
    if (n_test == 0u)                    return RF_ERROR_INVALID_PARAM;

    /* ---- Check CUDA device availability ---- */
    int dev_count = 0;
    if (!cuda_check(cudaGetDeviceCount(&dev_count), "cudaGetDeviceCount") ||
        dev_count < 1) {
        std::fprintf(stderr, "[rf_cuda] No CUDA device; falling back to CPU.\n");
        return rf_predict(handle, X_test, n_test, out_pred, out_tree_preds);
    }

    const int n_trees = static_cast<int>(rf_get_n_estimators(handle));
    const size_t dim  = handle->dim;

    /* ---- Serialise forest into flat GPU-friendly arrays (host side) ---- */
    std::vector<RfNodeGpu>  h_nodes;
    std::vector<int>        h_offsets(n_trees + 1);

    const size_t total_nodes = rf_get_n_nodes_total(handle);
    h_nodes.resize(total_nodes);
    size_t write_pos = 0u;

    for (int t = 0; t < n_trees; ++t) {
        h_offsets[t] = static_cast<int>(write_pos);
        for (const auto& nd : handle->trees[t].nodes) {
            h_nodes[write_pos++] = RfNodeGpu{
                nd.feature_idx,
                nd.left,
                nd.right,
                nd.threshold,
                nd.leaf_value
            };
        }
    }
    h_offsets[n_trees] = static_cast<int>(write_pos);

    /* ---- Allocate device memory ---- */
    RfNodeGpu* d_nodes          = nullptr;
    int*       d_tree_offsets   = nullptr;
    double*    d_X_test         = nullptr;
    double*    d_out_pred       = nullptr;
    double*    d_out_tree_preds = nullptr;

    const size_t bytes_nodes   = total_nodes * sizeof(RfNodeGpu);
    const size_t bytes_offsets = (n_trees + 1) * sizeof(int);
    const size_t bytes_X       = n_test * dim  * sizeof(double);
    const size_t bytes_pred    = n_test         * sizeof(double);
    const size_t bytes_tp      = out_tree_preds
                                 ? static_cast<size_t>(n_trees) * n_test * sizeof(double)
                                 : 0u;

    bool ok = true;
    ok = ok && cuda_check(cudaMalloc(&d_nodes,        bytes_nodes),   "malloc nodes");
    ok = ok && cuda_check(cudaMalloc(&d_tree_offsets, bytes_offsets), "malloc offsets");
    ok = ok && cuda_check(cudaMalloc(&d_X_test,       bytes_X),       "malloc X_test");
    ok = ok && cuda_check(cudaMalloc(&d_out_pred,     bytes_pred),    "malloc out_pred");
    if (out_tree_preds && ok) {
        ok = ok && cuda_check(cudaMalloc(&d_out_tree_preds, bytes_tp), "malloc tree_preds");
    }

    if (!ok) goto cuda_fail;

    /* ---- Copy to device ---- */
    ok = ok && cuda_check(
        cudaMemcpy(d_nodes, h_nodes.data(), bytes_nodes, cudaMemcpyHostToDevice),
        "memcpy nodes");
    ok = ok && cuda_check(
        cudaMemcpy(d_tree_offsets, h_offsets.data(), bytes_offsets, cudaMemcpyHostToDevice),
        "memcpy offsets");
    ok = ok && cuda_check(
        cudaMemcpy(d_X_test, X_test, bytes_X, cudaMemcpyHostToDevice),
        "memcpy X_test");

    if (!ok) goto cuda_fail;

    /* ---- Launch kernel ---- */
    {
        const int block_size = (n_trees < kMaxTPB) ? n_trees : kMaxTPB;
        const int grid_size  = static_cast<int>(n_test);
        const size_t smem    = sizeof(double); /* one shared accumulator   */

        rf_inference_kernel<<<grid_size, block_size, smem>>>(
            d_nodes, d_tree_offsets, n_trees,
            d_X_test, n_test, dim,
            d_out_pred, d_out_tree_preds
        );

        ok = ok && cuda_check(cudaGetLastError(),      "kernel launch");
        ok = ok && cuda_check(cudaDeviceSynchronize(), "device sync");
    }

    if (!ok) goto cuda_fail;

    /* ---- Copy results back ---- */
    ok = ok && cuda_check(
        cudaMemcpy(out_pred, d_out_pred, bytes_pred, cudaMemcpyDeviceToHost),
        "memcpy out_pred D->H");
    if (out_tree_preds && ok) {
        ok = ok && cuda_check(
            cudaMemcpy(out_tree_preds, d_out_tree_preds, bytes_tp,
                       cudaMemcpyDeviceToHost),
            "memcpy tree_preds D->H");
    }

    if (!ok) goto cuda_fail;

    /* ---- Cleanup and return ---- */
    cudaFree(d_nodes);
    cudaFree(d_tree_offsets);
    cudaFree(d_X_test);
    cudaFree(d_out_pred);
    if (d_out_tree_preds) cudaFree(d_out_tree_preds);
    return RF_SUCCESS;

cuda_fail:
    cudaFree(d_nodes);
    cudaFree(d_tree_offsets);
    cudaFree(d_X_test);
    cudaFree(d_out_pred);
    if (d_out_tree_preds) cudaFree(d_out_tree_preds);

    std::fprintf(stderr,
                 "[rf_cuda] CUDA inference failed — falling back to CPU.\n");
    return rf_predict(handle, X_test, n_test, out_pred, out_tree_preds);
}

/* ======================================================================
 * End of CUDA path
 * ====================================================================== */
#else  /* !ENABLE_CUDA — pure CPU stubs */

extern "C"
RfStatusCode rf_predict_cuda(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
) {
    /* CUDA is not compiled in; delegate to the CPU implementation.        */
    return rf_predict(handle, X_test, n_test, out_pred, out_tree_preds);
}

#endif /* ENABLE_CUDA */
