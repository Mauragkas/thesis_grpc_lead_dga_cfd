#include "gp_backend.h"
#include "gp_cpu.h"
#include "gp_cuda.h"
#include "gp_rocm.h"
#include <cstdlib>
#include <new>

struct GpBackendHandle {
    GpDeviceType device_type;
    int device_id;
};

GpStatusCode gp_backend_create(GpDeviceType device_type, int device_id, GpBackendHandle** out_handle) {
    if (!out_handle) return GP_ERROR_NULL_POINTER;

    GpBackendHandle* handle = new (std::nothrow) GpBackendHandle();
    if (!handle) return GP_ERROR_UNKNOWN;

    handle->device_type = device_type;
    handle->device_id = device_id;

    *out_handle = handle;
    return GP_SUCCESS;
}

void gp_backend_free(GpBackendHandle* handle) {
    if (handle) {
        delete handle;
    }
}

GpDeviceType gp_backend_get_type(const GpBackendHandle* handle) {
    if (!handle) return GP_DEVICE_CPU;
    return handle->device_type;
}

GpStatusCode gp_compute_covariance(
    GpBackendHandle* handle,
    const double* X1,
    size_t n1,
    const double* X2,
    size_t n2,
    size_t dim,
    const GpHyperparams* params,
    bool add_diagonal_noise,
    double* out_K
) {
    if (!handle) return GP_ERROR_NULL_POINTER;

    if (handle->device_type == GP_DEVICE_CUDA) {
        GpStatusCode status = cuda_compute_covariance(
            handle->device_id, X1, n1, X2, n2, dim, params, add_diagonal_noise, out_K
        );
        if (status == GP_SUCCESS) return GP_SUCCESS;
        /* Fallback to CPU if CUDA execution failed */
    } else if (handle->device_type == GP_DEVICE_ROCM) {
        GpStatusCode status = rocm_compute_covariance(
            handle->device_id, X1, n1, X2, n2, dim, params, add_diagonal_noise, out_K
        );
        if (status == GP_SUCCESS) return GP_SUCCESS;
        /* Fallback to CPU if ROCm execution failed */
    }

    return cpu_compute_covariance(X1, n1, X2, n2, dim, params, add_diagonal_noise, out_K);
}

GpStatusCode gp_cholesky(
    GpBackendHandle* handle,
    const double* K,
    size_t n,
    double* out_L
) {
    if (!handle) return GP_ERROR_NULL_POINTER;

    if (handle->device_type == GP_DEVICE_CUDA) {
        GpStatusCode status = cuda_cholesky(handle->device_id, K, n, out_L);
        if (status == GP_SUCCESS) return GP_SUCCESS;
    } else if (handle->device_type == GP_DEVICE_ROCM) {
        GpStatusCode status = rocm_cholesky(handle->device_id, K, n, out_L);
        if (status == GP_SUCCESS) return GP_SUCCESS;
    }

    return cpu_cholesky(K, n, out_L);
}

GpStatusCode gp_solve_triangular_lower(
    GpBackendHandle* handle,
    const double* L,
    const double* b,
    size_t n,
    double* out_y
) {
    (void)handle;
    return cpu_solve_triangular_lower(L, b, n, out_y);
}

GpStatusCode gp_solve_triangular_lower_transpose(
    GpBackendHandle* handle,
    const double* L,
    const double* y,
    size_t n,
    double* out_x
) {
    (void)handle;
    return cpu_solve_triangular_lower_transpose(L, y, n, out_x);
}

GpStatusCode gp_compute_alpha(
    GpBackendHandle* handle,
    const double* L,
    const double* y,
    size_t n,
    double* out_alpha
) {
    (void)handle;
    return cpu_compute_alpha(L, y, n, out_alpha);
}

GpStatusCode gp_predict_batch(
    GpBackendHandle* handle,
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
    if (!handle) return GP_ERROR_NULL_POINTER;

    if (handle->device_type == GP_DEVICE_CUDA) {
        GpStatusCode status = cuda_predict_batch(
            handle->device_id, X_train, n_train, X_test, n_test, dim, alpha, L, params, out_mean, out_var
        );
        if (status == GP_SUCCESS) return GP_SUCCESS;
    } else if (handle->device_type == GP_DEVICE_ROCM) {
        GpStatusCode status = rocm_predict_batch(
            handle->device_id, X_train, n_train, X_test, n_test, dim, alpha, L, params, out_mean, out_var
        );
        if (status == GP_SUCCESS) return GP_SUCCESS;
    }

    return cpu_predict_batch(X_train, n_train, X_test, n_test, dim, alpha, L, params, out_mean, out_var);
}

GpStatusCode gp_log_marginal_likelihood(
    GpBackendHandle* handle,
    const double* y,
    const double* alpha,
    const double* L,
    size_t n,
    double* out_mll
) {
    (void)handle;
    return cpu_log_marginal_likelihood(y, alpha, L, n, out_mll);
}
