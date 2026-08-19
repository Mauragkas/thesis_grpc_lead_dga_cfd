#include "gp_cpu.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline double compute_distance_sq(
    const double* x1,
    const double* x2,
    size_t dim,
    const double* length_scales
) {
    double sum = 0.0;
    for (size_t d = 0; d < dim; ++d) {
        double diff = (x1[d] - x2[d]) / length_scales[d];
        sum += diff * diff;
    }
    return sum;
}

inline double eval_kernel_scalar(
    double dist_sq,
    GpKernelType kernel_type,
    double signal_variance
) {
    if (kernel_type == GP_KERNEL_MATERN52) {
        double dist = std::sqrt(std::max(0.0, dist_sq));
        double sqrt5_r = std::sqrt(5.0) * dist;
        return signal_variance * (1.0 + sqrt5_r + (5.0 / 3.0) * dist_sq) * std::exp(-sqrt5_r);
    } else {
        /* GP_KERNEL_RBF */
        return signal_variance * std::exp(-0.5 * dist_sq);
    }
}

} // namespace

GpStatusCode cpu_compute_covariance(
    const double* X1,
    size_t n1,
    const double* X2,
    size_t n2,
    size_t dim,
    const GpHyperparams* params,
    bool add_diagonal_noise,
    double* out_K
) {
    if (!X1 || !X2 || !params || !out_K || !params->length_scales) {
        return GP_ERROR_NULL_POINTER;
    }
    if (dim == 0 || n1 == 0 || n2 == 0 || params->num_dim != dim) {
        return GP_ERROR_DIMENSION_MISMATCH;
    }

    const double* ls = params->length_scales;
    const double sig_var = params->signal_variance;
    const double noise_var = params->noise_variance;
    const GpKernelType ktype = params->kernel_type;
    const bool is_self_cov = (X1 == X2 && n1 == n2);

    if (is_self_cov) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (size_t i = 0; i < n1; ++i) {
            const double* x1 = &X1[i * dim];
            for (size_t j = i; j < n1; ++j) {
                const double* x2 = &X1[j * dim];
                double d2 = compute_distance_sq(x1, x2, dim, ls);
                double k_val = eval_kernel_scalar(d2, ktype, sig_var);
                if (i == j && add_diagonal_noise) {
                    k_val += noise_var;
                }
                out_K[i * n1 + j] = k_val;
                out_K[j * n1 + i] = k_val;
            }
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                const double* x1 = &X1[i * dim];
                const double* x2 = &X2[j * dim];
                double d2 = compute_distance_sq(x1, x2, dim, ls);
                out_K[i * n2 + j] = eval_kernel_scalar(d2, ktype, sig_var);
            }
        }
    }

    return GP_SUCCESS;
}

GpStatusCode cpu_cholesky(
    const double* K,
    size_t n,
    double* out_L
) {
    if (!K || !out_L) return GP_ERROR_NULL_POINTER;
    if (n == 0) return GP_ERROR_DIMENSION_MISMATCH;

    std::memset(out_L, 0, n * n * sizeof(double));

    for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            double val = out_L[j * n + k];
            sum += val * val;
        }

        double diag = K[j * n + j] - sum;
        if (diag <= 1e-14 || std::isnan(diag)) {
            return GP_ERROR_NOT_POSITIVE_DEFINITE;
        }

        double l_jj = std::sqrt(diag);
        out_L[j * n + j] = l_jj;
        double inv_l_jj = 1.0 / l_jj;

        #pragma omp parallel for schedule(static, 32)
        for (size_t i = j + 1; i < n; ++i) {
            double dot = 0.0;
            for (size_t k = 0; k < j; ++k) {
                dot += out_L[i * n + k] * out_L[j * n + k];
            }
            out_L[i * n + j] = (K[i * n + j] - dot) * inv_l_jj;
        }
    }

    return GP_SUCCESS;
}

GpStatusCode cpu_solve_triangular_lower(
    const double* L,
    const double* b,
    size_t n,
    double* out_y
) {
    if (!L || !b || !out_y) return GP_ERROR_NULL_POINTER;
    if (n == 0) return GP_ERROR_DIMENSION_MISMATCH;

    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t k = 0; k < i; ++k) {
            sum += L[i * n + k] * out_y[k];
        }
        double l_ii = L[i * n + i];
        if (std::abs(l_ii) < 1e-15) return GP_ERROR_NUMERICAL;
        out_y[i] = (b[i] - sum) / l_ii;
    }

    return GP_SUCCESS;
}

GpStatusCode cpu_solve_triangular_lower_transpose(
    const double* L,
    const double* y,
    size_t n,
    double* out_x
) {
    if (!L || !y || !out_x) return GP_ERROR_NULL_POINTER;
    if (n == 0) return GP_ERROR_DIMENSION_MISMATCH;

    for (size_t idx = 0; idx < n; ++idx) {
        size_t i = n - 1 - idx;
        double sum = 0.0;
        for (size_t k = i + 1; k < n; ++k) {
            sum += L[k * n + i] * out_x[k];
        }
        double l_ii = L[i * n + i];
        if (std::abs(l_ii) < 1e-15) return GP_ERROR_NUMERICAL;
        out_x[i] = (y[i] - sum) / l_ii;
    }

    return GP_SUCCESS;
}

GpStatusCode cpu_compute_alpha(
    const double* L,
    const double* y,
    size_t n,
    double* out_alpha
) {
    if (!L || !y || !out_alpha) return GP_ERROR_NULL_POINTER;
    if (n == 0) return GP_ERROR_DIMENSION_MISMATCH;

    std::vector<double> temp(n);
    GpStatusCode status = cpu_solve_triangular_lower(L, y, n, temp.data());
    if (status != GP_SUCCESS) return status;

    return cpu_solve_triangular_lower_transpose(L, temp.data(), n, out_alpha);
}

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
) {
    if (!X_train || !X_test || !alpha || !params || !out_mean) {
        return GP_ERROR_NULL_POINTER;
    }
    if (n_train == 0 || n_test == 0 || dim == 0 || params->num_dim != dim) {
        return GP_ERROR_DIMENSION_MISMATCH;
    }

    const double* ls = params->length_scales;
    const double sig_var = params->signal_variance;
    const GpKernelType ktype = params->kernel_type;

    #pragma omp parallel for schedule(dynamic, 8)
    for (size_t m = 0; m < n_test; ++m) {
        const double* x_test_m = &X_test[m * dim];

        double mean_val = 0.0;
        std::vector<double> k_m(n_train);

        for (size_t i = 0; i < n_train; ++i) {
            const double* x_train_i = &X_train[i * dim];
            double d2 = compute_distance_sq(x_test_m, x_train_i, dim, ls);
            double k_val = eval_kernel_scalar(d2, ktype, sig_var);
            k_m[i] = k_val;
            mean_val += k_val * alpha[i];
        }
        out_mean[m] = mean_val;

        if (out_var && L) {
            std::vector<double> v(n_train);
            /* Forward solve: L * v = k_m */
            for (size_t i = 0; i < n_train; ++i) {
                double s = 0.0;
                for (size_t k = 0; k < i; ++k) {
                    s += L[i * n_train + k] * v[k];
                }
                v[i] = (k_m[i] - s) / L[i * n_train + i];
            }

            double v_dot = 0.0;
            for (size_t i = 0; i < n_train; ++i) {
                v_dot += v[i] * v[i];
            }

            double prior_var = sig_var;
            out_var[m] = std::max(0.0, prior_var - v_dot);
        }
    }

    return GP_SUCCESS;
}

GpStatusCode cpu_log_marginal_likelihood(
    const double* y,
    const double* alpha,
    const double* L,
    size_t n,
    double* out_mll
) {
    if (!y || !alpha || !L || !out_mll) return GP_ERROR_NULL_POINTER;
    if (n == 0) return GP_ERROR_DIMENSION_MISMATCH;

    double y_dot_alpha = 0.0;
    for (size_t i = 0; i < n; ++i) {
        y_dot_alpha += y[i] * alpha[i];
    }

    double log_det = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double l_ii = L[i * n + i];
        if (l_ii <= 0.0 || std::isnan(l_ii)) return GP_ERROR_NUMERICAL;
        log_det += std::log(l_ii);
    }

    const double kPi = 3.14159265358979323846;
    double mll = -0.5 * y_dot_alpha - log_det - 0.5 * static_cast<double>(n) * std::log(2.0 * kPi);
    *out_mll = mll;

    return GP_SUCCESS;
}
