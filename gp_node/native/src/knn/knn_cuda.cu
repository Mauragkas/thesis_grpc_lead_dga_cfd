/*
 * knn_cuda.cu — CUDA implementation of knn_predict_cuda.
 *
 * Strategy (ENABLE_CUDA build):
 *   - Copy X_test and model's X_train + y_train + length_scales to device.
 *   - Launch knn_kernel<<<n_test, block_size>>> where each block handles one
 *     test point cooperatively.  Threads within the block each compute a
 *     contiguous slice of training distances, then a shared-memory insertion
 *     sort maintains a per-block top-k heap (k is small, typically 7).
 *   - After synchronisation collect predictions and memcpy back to host.
 *   - On any CUDA error, print a warning and fall back to knn_predict (CPU).
 *
 * When ENABLE_CUDA is not defined the file compiles to CPU-fallback stubs so
 * that the project builds cleanly without nvcc.
 */

#include "knn_backend.h"
#include "knn_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CPU fallback — declared here, defined in knn_cpu.cpp. */
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

/* ================================================================== */
/* CUDA path                                                            */
/* ================================================================== */
#if defined(__CUDACC__) || defined(ENABLE_CUDA)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------
 * Device-side helpers
 * ------------------------------------------------------------------ */
namespace {

__device__ inline double d_scaled_dist_sq(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ ls,
    unsigned int dim
) {
    double sum = 0.0;
    for (unsigned int d = 0; d < dim; ++d) {
        double diff = (a[d] - b[d]) / ls[d];
        sum += diff * diff;
    }
    return sum;
}

/* ------------------------------------------------------------------
 * knn_kernel
 *
 * Grid  : n_test blocks  (one block per query point)
 * Block : min(n_train, 256) threads
 *
 * Shared memory layout (per block):
 *   double  sh_topk_d2[MAX_K]   — squared distances of top-k neighbours
 *   int     sh_topk_idx[MAX_K]  — their training indices
 *
 * k is constrained to <= MAX_K at launch time.
 * Each thread handles a stripe of training points; after the stripe loop
 * thread 0 accumulates the block's partial top-k via insertion sort,
 * then computes the IDW prediction.
 * ------------------------------------------------------------------ */
#define KNN_CUDA_MAX_K 32

__global__ void knn_kernel(
    const double* __restrict__ d_Xtr,  /* n_train * dim */
    const double* __restrict__ d_ytr,  /* n_train       */
    const double* __restrict__ d_Xte,  /* n_test  * dim */
    const double* __restrict__ d_ls,   /* dim           */
    unsigned int n_train,
    unsigned int dim,
    unsigned int k,
    double*      d_pred,               /* n_test        */
    double*      d_dist                /* n_test, may be NULL */
) {
    /* Each block is responsible for one test point. */
    const unsigned int m = blockIdx.x;
    const double* x_test_m = d_Xte + (size_t)m * dim;

    /* Per-thread partial top-k stored in registers / local memory. */
    __shared__ double sh_d2 [KNN_CUDA_MAX_K];
    __shared__ int    sh_idx[KNN_CUDA_MAX_K];

    /* Initialise shared top-k (only thread 0 does it). */
    if (threadIdx.x == 0) {
        for (unsigned int ki = 0; ki < k; ++ki) {
            sh_d2 [ki] = 1e300;
            sh_idx[ki] = -1;
        }
    }
    __syncthreads();

    /* Per-thread local top-k heap. */
    double loc_d2 [KNN_CUDA_MAX_K];
    int    loc_idx[KNN_CUDA_MAX_K];
    for (unsigned int ki = 0; ki < k; ++ki) {
        loc_d2 [ki] = 1e300;
        loc_idx[ki] = -1;
    }

    /* Each thread processes its stripe of training points. */
    for (unsigned int i = threadIdx.x; i < n_train; i += blockDim.x) {
        double d2 = d_scaled_dist_sq(x_test_m, d_Xtr + (size_t)i * dim, d_ls, dim);

        /* Insertion into the local top-k (ascending order). */
        if (d2 < loc_d2[k - 1]) {
            loc_d2 [k - 1] = d2;
            loc_idx[k - 1] = (int)i;
            /* Bubble the new entry into position. */
            for (int ki = (int)k - 2; ki >= 0; --ki) {
                if (loc_d2[ki + 1] < loc_d2[ki]) {
                    double td = loc_d2[ki]; loc_d2[ki] = loc_d2[ki+1]; loc_d2[ki+1] = td;
                    int    ti = loc_idx[ki]; loc_idx[ki] = loc_idx[ki+1]; loc_idx[ki+1] = ti;
                } else {
                    break;
                }
            }
        }
    }

    /* Merge per-thread local heaps into shared top-k one thread at a time. */
    __syncthreads();
    for (unsigned int t = 0; t < blockDim.x; ++t) {
        if (threadIdx.x == t) {
            for (unsigned int li = 0; li < k; ++li) {
                if (loc_d2[li] >= 1e299) break; /* rest are sentinel */
                double d2  = loc_d2[li];
                int    idx = loc_idx[li];
                if (d2 < sh_d2[k - 1]) {
                    sh_d2 [k - 1] = d2;
                    sh_idx[k - 1] = idx;
                    for (int ki = (int)k - 2; ki >= 0; --ki) {
                        if (sh_d2[ki + 1] < sh_d2[ki]) {
                            double td = sh_d2[ki]; sh_d2[ki] = sh_d2[ki+1]; sh_d2[ki+1] = td;
                            int    ti = sh_idx[ki]; sh_idx[ki] = sh_idx[ki+1]; sh_idx[ki+1] = ti;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    /* Thread 0 computes the IDW prediction from the shared top-k. */
    if (threadIdx.x == 0) {
        constexpr double kEps = 1e-12;
        double w_sum  = 0.0;
        double wy_sum = 0.0;
        double d_sum  = 0.0;

        for (unsigned int ki = 0; ki < k; ++ki) {
            int cidx = sh_idx[ki];
            if (cidx < 0) continue;
            double d  = sqrt(sh_d2[ki]);
            double w  = 1.0 / (d + kEps);
            w_sum  += w;
            wy_sum += w * d_ytr[cidx];
            d_sum  += d;
        }

        d_pred[m] = wy_sum / w_sum;
        if (d_dist) {
            d_dist[m] = d_sum / (double)k;
        }
    }
}

/* Helper: free a list of device pointers and return an error code. */
#define CUDA_CLEANUP_RETURN(code, ...)                        \
    do {                                                      \
        const double* _ptrs[] = { __VA_ARGS__ };              \
        for (auto p : _ptrs) {                                \
            if (p) cudaFree((void*)p);                        \
        }                                                     \
        return (code);                                        \
    } while (0)

} // anonymous namespace

/* ------------------------------------------------------------------
 * knn_predict_cuda — public entry point
 * ------------------------------------------------------------------ */
extern "C"
KnnStatusCode knn_predict_cuda(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
) {
    if (!handle || !X_test || !out_pred) return KNN_ERROR_NULL_POINTER;
    if (n_test == 0)                      return KNN_ERROR_DIMENSION_MISMATCH;

    const size_t   n_train = handle->n_train;
    const size_t   dim     = handle->dim;
    const uint32_t k       = handle->k;
    const double*  Xtr     = handle->X_train.data();
    const double*  ytr     = handle->y_train.data();
    const double*  ls      = handle->length_scales.data();

    if (k > KNN_CUDA_MAX_K) {
        /* k too large for shared-memory kernel; fall back to CPU. */
        fprintf(stderr, "[knn_cuda] k=%u > KNN_CUDA_MAX_K=%d; falling back to CPU.\n",
                k, KNN_CUDA_MAX_K);
        return knn_predict(handle, X_test, n_test, out_pred, out_dist);
    }

    /* Check device availability. */
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        fprintf(stderr, "[knn_cuda] No CUDA device found; falling back to CPU.\n");
        return knn_predict(handle, X_test, n_test, out_pred, out_dist);
    }

    /* Allocate device buffers. */
    double *d_Xtr = nullptr, *d_ytr = nullptr, *d_ls  = nullptr;
    double *d_Xte = nullptr;
    double *d_pred = nullptr, *d_dist_buf = nullptr;

    size_t bytes_Xtr  = n_train * dim  * sizeof(double);
    size_t bytes_ytr  = n_train        * sizeof(double);
    size_t bytes_ls   = dim            * sizeof(double);
    size_t bytes_Xte  = n_test  * dim  * sizeof(double);
    size_t bytes_pred = n_test         * sizeof(double);

    auto alloc_fail = [&]() -> KnnStatusCode {
        fprintf(stderr, "[knn_cuda] cudaMalloc failed; falling back to CPU.\n");
        if (d_Xtr)      cudaFree(d_Xtr);
        if (d_ytr)      cudaFree(d_ytr);
        if (d_ls)       cudaFree(d_ls);
        if (d_Xte)      cudaFree(d_Xte);
        if (d_pred)     cudaFree(d_pred);
        if (d_dist_buf) cudaFree(d_dist_buf);
        return knn_predict(handle, X_test, n_test, out_pred, out_dist);
    };

    if (cudaMalloc(&d_Xtr,  bytes_Xtr)  != cudaSuccess) return alloc_fail();
    if (cudaMalloc(&d_ytr,  bytes_ytr)  != cudaSuccess) return alloc_fail();
    if (cudaMalloc(&d_ls,   bytes_ls)   != cudaSuccess) return alloc_fail();
    if (cudaMalloc(&d_Xte,  bytes_Xte)  != cudaSuccess) return alloc_fail();
    if (cudaMalloc(&d_pred, bytes_pred) != cudaSuccess) return alloc_fail();
    if (out_dist) {
        if (cudaMalloc(&d_dist_buf, bytes_pred) != cudaSuccess) return alloc_fail();
    }

    /* Copy inputs to device. */
    cudaMemcpy(d_Xtr, Xtr,    bytes_Xtr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ytr, ytr,    bytes_ytr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls,  ls,     bytes_ls,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xte, X_test, bytes_Xte, cudaMemcpyHostToDevice);

    /* Launch: one block per test point; threads cooperate over training set. */
    unsigned int block_sz = (unsigned int)std::min(n_train, (size_t)256);
    dim3 grid(static_cast<unsigned int>(n_test));
    dim3 block(block_sz);

    knn_kernel<<<grid, block>>>(
        d_Xtr, d_ytr, d_Xte, d_ls,
        static_cast<unsigned int>(n_train),
        static_cast<unsigned int>(dim),
        k,
        d_pred,
        d_dist_buf
    );

    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[knn_cuda] kernel error: %s — falling back to CPU.\n",
                cudaGetErrorString(kerr));
        cudaFree(d_Xtr); cudaFree(d_ytr); cudaFree(d_ls);
        cudaFree(d_Xte); cudaFree(d_pred);
        if (d_dist_buf) cudaFree(d_dist_buf);
        return knn_predict(handle, X_test, n_test, out_pred, out_dist);
    }

    cudaDeviceSynchronize();

    /* Copy results back. */
    cudaMemcpy(out_pred, d_pred, bytes_pred, cudaMemcpyDeviceToHost);
    if (out_dist && d_dist_buf) {
        cudaMemcpy(out_dist, d_dist_buf, bytes_pred, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_Xtr); cudaFree(d_ytr); cudaFree(d_ls);
    cudaFree(d_Xte); cudaFree(d_pred);
    if (d_dist_buf) cudaFree(d_dist_buf);

    return KNN_SUCCESS;
}

/* ================================================================== */
/* Stub path — compiled when ENABLE_CUDA / __CUDACC__ is not defined   */
/* ================================================================== */
#else /* !ENABLE_CUDA */

#include <cstdio>

extern "C"
KnnStatusCode knn_predict_cuda(
    const KnnModelHandle* handle,
    const double*         X_test,
    size_t                n_test,
    double*               out_pred,
    double*               out_dist
) {
    /* CUDA runtime not linked; fall back transparently to CPU path. */
    return knn_predict(handle, X_test, n_test, out_pred, out_dist);
}

#endif /* ENABLE_CUDA */
