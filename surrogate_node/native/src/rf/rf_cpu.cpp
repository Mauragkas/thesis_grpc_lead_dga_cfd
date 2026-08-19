/*
 * rf_cpu.cpp — CPU-side Random Forest regression backend.
 *
 * Training:  OpenMP-parallel, one thread per tree.
 * Inference: OpenMP-parallel over test points.
 *
 * rf_predict_cuda / rf_predict_rocm in this file are CPU fallbacks only;
 * the real CUDA path lives in rf_cuda.cu and the ROCm path in rf_rocm.cpp.
 * The build system should compile exactly one of those two translation
 * units and link it instead of the stubs here.
 */

#include "rf_backend.h"
#include "rf_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* ======================================================================
 * File-local types and helpers (anonymous namespace)
 * ====================================================================== */
namespace {

/* ---------------------------------------------------------------------- */
/* Compute mean and total sum-of-squared-deviations for y[indices[begin..end)].
 * Returns {mean, mse_total}.                                              */
/* ---------------------------------------------------------------------- */
inline std::pair<double, double> subset_mse(
    const double*              y,
    const std::vector<size_t>& indices,
    size_t                     begin,
    size_t                     end
) {
    if (end <= begin) return {0.0, 0.0};
    const size_t n = end - begin;
    double sum = 0.0;
    for (size_t k = begin; k < end; ++k) sum += y[indices[k]];
    const double mean = sum / static_cast<double>(n);
    double mse = 0.0;
    for (size_t k = begin; k < end; ++k) {
        double d = y[indices[k]] - mean;
        mse += d * d;
    }
    return {mean, mse};
}

/* ---------------------------------------------------------------------- */
/* Recursive tree builder.
 *
 * X_boot  — bootstrap feature matrix (n_boot x dim, row-major)
 * y_boot  — bootstrap target vector  (n_boot)
 * idx     — working sample-index buffer; [begin,end) are this node's samples
 * depth   — current recursion depth (root = 0)
 * params  — hyperparameters (read-only)
 * dim     — feature dimension
 * rng     — per-tree Mersenne Twister (caller must not share across threads)
 * nodes   — output node vector (appended to in place)
 *
 * Returns the index within `nodes` of the root node created here.        */
/* ---------------------------------------------------------------------- */
int build_tree(
    const double*              X_boot,
    const double*              y_boot,
    std::vector<size_t>&       idx,
    size_t                     begin,
    size_t                     end,
    int                        depth,
    const RfHyperparams&       params,
    size_t                     dim,
    std::mt19937_64&           rng,
    std::vector<RfNode>&       nodes
) {
    const size_t n_samples  = end - begin;
    const uint32_t max_depth = params.max_depth;
    const uint32_t min_leaf  = params.min_samples_leaf < 1u ? 1u : params.min_samples_leaf;

    /* ---- Compute current node stats ---- */
    auto [cur_mean, cur_mse] = subset_mse(y_boot, idx, begin, end);

    /* ---- Leaf conditions ---- */
    const bool depth_stop = (max_depth > 0u &&
                             static_cast<uint32_t>(depth) >= max_depth);
    const bool size_stop  = (n_samples < 2u * static_cast<size_t>(min_leaf));

    auto make_leaf = [&]() -> int {
        RfNode leaf{-1, 0.0, cur_mean, -1, -1};
        nodes.push_back(leaf);
        return static_cast<int>(nodes.size()) - 1;
    };

    if (depth_stop || size_stop || n_samples == 0u) {
        return make_leaf();
    }

    /* ---- Random feature subset selection (partial Fisher-Yates) ---- */
    uint32_t max_feat = params.max_features;
    if (max_feat == 0u || max_feat > static_cast<uint32_t>(dim)) {
        max_feat = static_cast<uint32_t>(dim);
    }
    std::vector<size_t> feat_pool(dim);
    std::iota(feat_pool.begin(), feat_pool.end(), static_cast<size_t>(0));
    for (uint32_t f = 0u; f < max_feat; ++f) {
        std::uniform_int_distribution<size_t> uid(f, dim - 1u);
        std::swap(feat_pool[f], feat_pool[uid(rng)]);
    }

    /* ---- Split search ---- */
    constexpr size_t kMaxThresholds = 32u;
    double best_gain      = -1.0;
    int    best_feature   = -1;
    double best_threshold = 0.0;

    /* Sort buffer: local copy of current sample indices for feature sorting */
    std::vector<size_t> sort_buf(idx.begin() + begin, idx.begin() + end);

    for (uint32_t fi = 0u; fi < max_feat; ++fi) {
        const size_t feat = feat_pool[fi];

        /* Sort samples by feature value */
        std::sort(sort_buf.begin(), sort_buf.end(),
                  [&](size_t a, size_t b) noexcept {
                      return X_boot[a * dim + feat] < X_boot[b * dim + feat];
                  });

        /* Collect midpoint thresholds between consecutive distinct values  */
        std::vector<double> thresholds;
        thresholds.reserve(n_samples - 1u);
        for (size_t k = 0u; k + 1u < n_samples; ++k) {
            const double v0 = X_boot[sort_buf[k]       * dim + feat];
            const double v1 = X_boot[sort_buf[k + 1u]  * dim + feat];
            if (v1 > v0 + 1e-15) {
                thresholds.push_back(0.5 * (v0 + v1));
            }
        }

        /* Sub-sample thresholds if too many; re-sort for incremental scan  */
        if (thresholds.size() > kMaxThresholds) {
            std::shuffle(thresholds.begin(), thresholds.end(), rng);
            thresholds.resize(kMaxThresholds);
            std::sort(thresholds.begin(), thresholds.end());
        }
        if (thresholds.empty()) continue;

        /*
         * Evaluate each threshold using incremental left/right accumulators.
         * mse(S) = sum_y2 - sum_y^2 / n   (equivalent to n * variance)
         */
        double r_sum  = 0.0, r_sum2 = 0.0;
        size_t r_n    = n_samples;
        double l_sum  = 0.0, l_sum2 = 0.0;
        size_t l_n    = 0u;

        for (size_t k = 0u; k < n_samples; ++k) {
            const double yv = y_boot[sort_buf[k]];
            r_sum  += yv;
            r_sum2 += yv * yv;
        }

        size_t ptr = 0u;
        for (const double thr : thresholds) {
            /* Move samples with x[feat] <= thr to left side                */
            while (ptr < n_samples &&
                   X_boot[sort_buf[ptr] * dim + feat] <= thr) {
                const double yv = y_boot[sort_buf[ptr++]];
                l_sum += yv;  l_sum2 += yv * yv;  ++l_n;
                r_sum -= yv;  r_sum2 -= yv * yv;  --r_n;
            }
            if (l_n < static_cast<size_t>(min_leaf) ||
                r_n < static_cast<size_t>(min_leaf)) continue;

            const double dl = static_cast<double>(l_n);
            const double dr = static_cast<double>(r_n);
            const double mse_l = l_sum2 - (l_sum * l_sum) / dl;
            const double mse_r = r_sum2 - (r_sum * r_sum) / dr;
            const double dn    = static_cast<double>(n_samples);
            const double gain  = cur_mse - (mse_l * dl + mse_r * dr) / dn;

            if (gain > best_gain) {
                best_gain      = gain;
                best_feature   = static_cast<int>(feat);
                best_threshold = thr;
            }
        }
    }

    /* ---- No valid split found — make leaf ---- */
    if (best_feature < 0 || best_gain <= 0.0) {
        return make_leaf();
    }

    /* ---- Partition idx[begin,end) on the best split ---- */
    const size_t bf = static_cast<size_t>(best_feature);
    auto it  = std::partition(idx.begin() + begin, idx.begin() + end,
                              [&](size_t si) noexcept {
                                  return X_boot[si * dim + bf] <= best_threshold;
                              });
    const size_t mid = static_cast<size_t>(it - idx.begin());

    /* Allocate the internal node slot; children will be appended after    */
    const int node_idx = static_cast<int>(nodes.size());
    nodes.push_back({}); /* placeholder filled in below                    */

    const int lc = build_tree(X_boot, y_boot, idx, begin, mid,
                               depth + 1, params, dim, rng, nodes);
    const int rc = build_tree(X_boot, y_boot, idx, mid,   end,
                               depth + 1, params, dim, rng, nodes);

    nodes[node_idx] = RfNode{best_feature, best_threshold, 0.0, lc, rc};
    return node_idx;
}

} /* anonymous namespace */

/* ======================================================================
 * Public C-ABI implementation
 * ====================================================================== */

/* ---------------------------------------------------------------------- */
/* rf_model_train                                                           */
/* ---------------------------------------------------------------------- */
RfStatusCode rf_model_train(
    const double*        X_train,
    const double*        y_train,
    size_t               n_train,
    size_t               dim,
    const RfHyperparams* params,
    RfModelHandle**      out_handle
) {
    if (!X_train || !y_train || !params || !out_handle) return RF_ERROR_NULL_POINTER;
    if (n_train == 0u || dim == 0u)                      return RF_ERROR_INVALID_PARAM;
    if (params->n_estimators == 0u)                      return RF_ERROR_INVALID_PARAM;

    RfModelHandle* handle = nullptr;
    try {
        handle = new RfModelHandle();
    } catch (...) {
        return RF_ERROR_ALLOC_FAIL;
    }
    handle->dim    = dim;
    handle->params = *params;
    handle->trees.resize(params->n_estimators); /* pre-size so indexed access is safe */

    const uint32_t n_est     = params->n_estimators;
    const uint64_t base_seed = params->seed;

    /*
     * Parallelise over trees.  schedule(dynamic,1) because trees have
     * variable build cost (deeper splits take longer).
     */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ti = 0; ti < static_cast<int>(n_est); ++ti) {
        std::mt19937_64 rng(base_seed + static_cast<uint64_t>(ti));

        /* ---- Bootstrap sample: n_train rows drawn with replacement ---- */
        std::uniform_int_distribution<size_t> uid(0u, n_train - 1u);
        std::vector<double> X_boot(n_train * dim);
        std::vector<double> y_boot(n_train);
        for (size_t s = 0u; s < n_train; ++s) {
            const size_t src = uid(rng);
            std::memcpy(&X_boot[s * dim], &X_train[src * dim],
                        dim * sizeof(double));
            y_boot[s] = y_train[src];
        }

        /* ---- Build tree ---- */
        std::vector<size_t> indices(n_train);
        std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));

        std::vector<RfNode> nodes;
        nodes.reserve(2u * n_train + 1u);

        build_tree(X_boot.data(), y_boot.data(), indices,
                   0u, n_train, 0, *params, dim, rng, nodes);

        /* ---- Store (critical section to avoid concurrent vector mutation) */
        #pragma omp critical
        {
            handle->trees[ti].nodes = std::move(nodes);
        }
    }

    *out_handle = handle;
    return RF_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* rf_model_free                                                            */
/* ---------------------------------------------------------------------- */
void rf_model_free(RfModelHandle* handle) {
    delete handle; /* safe if nullptr                                       */
}

/* ---------------------------------------------------------------------- */
/* rf_predict — CPU inference, OpenMP over test points                     */
/* ---------------------------------------------------------------------- */
RfStatusCode rf_predict(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds  /* may be NULL; layout [n_trees x n_test] */
) {
    if (!handle || !X_test || !out_pred) return RF_ERROR_NULL_POINTER;
    if (n_test == 0u)                    return RF_ERROR_INVALID_PARAM;

    const size_t n_trees     = handle->trees.size();
    const size_t dim         = handle->dim;
    const double inv_n_trees = 1.0 / static_cast<double>(n_trees);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0u; i < n_test; ++i) {
        const double* x = &X_test[i * dim];
        double acc = 0.0;
        for (size_t t = 0u; t < n_trees; ++t) {
            const double pred_t = handle->trees[t].predict(x, dim);
            acc += pred_t;
            if (out_tree_preds) {
                out_tree_preds[t * n_test + i] = pred_t;
            }
        }
        out_pred[i] = acc * inv_n_trees;
    }

    return RF_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* rf_predict_cuda — CPU fallback                                           */
/* Compiled only when the real CUDA TU (rf_cuda.cu) is NOT linked.         */
/* ---------------------------------------------------------------------- */
#ifndef ENABLE_CUDA
RfStatusCode rf_predict_cuda(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
) {
    return rf_predict(handle, X_test, n_test, out_pred, out_tree_preds);
}
#endif /* ENABLE_CUDA */

/* ---------------------------------------------------------------------- */
/* rf_predict_rocm — CPU fallback (overridden when rf_rocm.cpp is linked)  */
/* ---------------------------------------------------------------------- */
RfStatusCode rf_predict_rocm(
    const RfModelHandle* handle,
    const double*        X_test,
    size_t               n_test,
    double*              out_pred,
    double*              out_tree_preds
) {
    return rf_predict(handle, X_test, n_test, out_pred, out_tree_preds);
}

/* ---------------------------------------------------------------------- */
/* Metadata getters                                                         */
/* ---------------------------------------------------------------------- */
uint32_t rf_get_n_estimators(const RfModelHandle* handle) {
    if (!handle) return 0u;
    return static_cast<uint32_t>(handle->trees.size());
}

uint32_t rf_get_max_depth(const RfModelHandle* handle) {
    if (!handle) return 0u;
    return handle->params.max_depth;
}

size_t rf_get_n_nodes_total(const RfModelHandle* handle) {
    if (!handle) return 0u;
    size_t total = 0u;
    for (const auto& t : handle->trees) total += t.nodes.size();
    return total;
}
