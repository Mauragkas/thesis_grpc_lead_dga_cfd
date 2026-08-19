#ifndef GP_CUDA_H
#define GP_CUDA_H

#include "gp_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_is_available(int* out_device_count);

GpStatusCode cuda_compute_covariance(
    int device_id,
    const double* X1,
    size_t n1,
    const double* X2,
    size_t n2,
    size_t dim,
    const GpHyperparams* params,
    bool add_diagonal_noise,
    double* out_K
);

GpStatusCode cuda_cholesky(
    int device_id,
    const double* K,
    size_t n,
    double* out_L
);

GpStatusCode cuda_predict_batch(
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
);

#ifdef __cplusplus
}
#endif

#endif /* GP_CUDA_H */
