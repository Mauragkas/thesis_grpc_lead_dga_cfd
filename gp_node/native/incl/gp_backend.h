#ifndef GP_BACKEND_H
#define GP_BACKEND_H

#include "gp_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpBackendHandle GpBackendHandle;

/**
 * Creates a compute backend instance for the specified device.
 */
GpStatusCode gp_backend_create(GpDeviceType device_type, int device_id, GpBackendHandle** out_handle);

/**
 * Destroys a compute backend instance.
 */
void gp_backend_free(GpBackendHandle* handle);

/**
 * Returns the device type associated with the backend handle.
 */
GpDeviceType gp_backend_get_type(const GpBackendHandle* handle);

/**
 * Computes pairwise covariance matrix between X1 (N1 x D) and X2 (N2 x D).
 * If is_symmetric is true (and X1 == X2), diagonal noise variance is added.
 * Matrix layout is contiguous row-major (size N1 * N2).
 */
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
);

/**
 * Computes the lower Cholesky decomposition L of positive-definite matrix K (N x N).
 * Out matrix L is row-major (N x N).
 */
GpStatusCode gp_cholesky(
    GpBackendHandle* handle,
    const double* K,
    size_t n,
    double* out_L
);

/**
 * Solves L * y = b for y, where L is lower triangular (N x N) and b is (N x 1).
 */
GpStatusCode gp_solve_triangular_lower(
    GpBackendHandle* handle,
    const double* L,
    const double* b,
    size_t n,
    double* out_y
);

/**
 * Solves L^T * x = y for x, where L is lower triangular (N x N) and y is (N x 1).
 */
GpStatusCode gp_solve_triangular_lower_transpose(
    GpBackendHandle* handle,
    const double* L,
    const double* y,
    size_t n,
    double* out_x
);

/**
 * Computes alpha weights vector: alpha = (K + sigma_n^2 * I)^(-1) * y.
 * Requires precomputed lower Cholesky factor L (N x N).
 */
GpStatusCode gp_compute_alpha(
    GpBackendHandle* handle,
    const double* L,
    const double* y,
    size_t n,
    double* out_alpha
);

/**
 * Predicts mean and variance for test points X_test (M x D) using training points X_train (N x D),
 * weights alpha (N x 1), and Cholesky factor L (N x N).
 * out_var can be NULL if variance is not needed.
 */
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
);

/**
 * Computes marginal log-likelihood (MLL):
 * -0.5 * y^T * alpha - sum(ln(L_ii)) - 0.5 * N * ln(2*pi)
 */
GpStatusCode gp_log_marginal_likelihood(
    GpBackendHandle* handle,
    const double* y,
    const double* alpha,
    const double* L,
    size_t n,
    double* out_mll
);

#ifdef __cplusplus
}
#endif

#endif /* GP_BACKEND_H */
