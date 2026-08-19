#include "gp_rocm.h"
#include "gp_cpu.h"

#if defined(__HIPCC__) || defined(ENABLE_ROCM)
#include <hip/hip_runtime.h>

bool rocm_is_available(int* out_device_count) {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count <= 0) {
        if (out_device_count) *out_device_count = 0;
        return false;
    }
    if (out_device_count) *out_device_count = count;
    return true;
}

GpStatusCode rocm_compute_covariance(
    int device_id,
    const double* X1,
    size_t n1,
    const double* X2,
    size_t n2,
    size_t dim,
    const GpHyperparams* params,
    bool add_diagonal_noise,
    double* out_K
) {
    /* Fallback to CPU OpenMP if ROCm runtime kernels are mapped */
    return cpu_compute_covariance(X1, n1, X2, n2, dim, params, add_diagonal_noise, out_K);
}

GpStatusCode rocm_cholesky(
    int device_id,
    const double* K,
    size_t n,
    double* out_L
) {
    return cpu_cholesky(K, n, out_L);
}

GpStatusCode rocm_predict_batch(
    int device_id,
    const double* X_train,
    size_t n_train,
    const double* X_test,
    size_t n_test,
    size_t dim,
    const double* alpha,
    const double* L,
    const GpHyperparams* params,
    double* out_mean,
    double* out_var
) {
    return cpu_predict_batch(X_train, n_train, X_test, n_test, dim, alpha, L, params, out_mean, out_var);
}

#else

bool rocm_is_available(int* out_device_count) {
    if (out_device_count) *out_device_count = 0;
    return false;
}

GpStatusCode rocm_compute_covariance(
    int, const double*, size_t, const double*, size_t, size_t, const GpHyperparams*, bool, double*
) {
    return GP_ERROR_ROCM_FAIL;
}

GpStatusCode rocm_cholesky(int, const double*, size_t, double*) {
    return GP_ERROR_ROCM_FAIL;
}

GpStatusCode rocm_predict_batch(
    int, const double*, size_t, const double*, size_t, size_t, const double*, const double*, const GpHyperparams*, double*, double*
) {
    return GP_ERROR_ROCM_FAIL;
}

#endif
