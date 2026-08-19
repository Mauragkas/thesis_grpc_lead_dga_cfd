#ifndef GP_CPU_H
#define GP_CPU_H

#include "gp_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

GpStatusCode cpu_compute_covariance(
    const double* X1,
    size_t n1,
    const double* X2,
    size_t n2,
    size_t dim,
    const GpHyperparams* params,
    bool add_diagonal_noise,
    double* out_K
);

GpStatusCode cpu_cholesky(
    const double* K,
    size_t n,
    double* out_L
);

GpStatusCode cpu_solve_triangular_lower(
    const double* L,
    const double* b,
    size_t n,
    double* out_y
);

GpStatusCode cpu_solve_triangular_lower_transpose(
    const double* L,
    const double* y,
    size_t n,
    double* out_x
);

GpStatusCode cpu_compute_alpha(
    const double* L,
    const double* y,
    size_t n,
    double* out_alpha
);

GpStatusCode cpu_predict_batch(
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

GpStatusCode cpu_log_marginal_likelihood(
    const double* y,
    const double* alpha,
    const double* L,
    size_t n,
    double* out_mll
);

#ifdef __cplusplus
}
#endif

#endif /* GP_CPU_H */
