#ifndef RF_INTERNAL_H
#define RF_INTERNAL_H

#include "rf_backend.h"
#include <vector>
#include <cstddef>
#include <cstdint>

struct RfNode {
    int    feature_idx; /* -1 => leaf node */
    double threshold;   /* split: go left if x[feature_idx] <= threshold */
    double leaf_value;  /* prediction value (valid only when feature_idx == -1) */
    int    left;        /* index into nodes[] of left child */
    int    right;       /* index into nodes[] of right child */
};

struct RfTree {
    std::vector<RfNode> nodes;

    double predict(const double* x, size_t /*dim*/) const {
        int idx = 0;
        while (nodes[idx].feature_idx != -1) {
            const RfNode& n = nodes[idx];
            idx = (x[n.feature_idx] <= n.threshold) ? n.left : n.right;
        }
        return nodes[idx].leaf_value;
    }
};

struct RfModelHandle {
    std::vector<RfTree> trees;
    size_t              dim;
    RfHyperparams       params;
};

#endif /* RF_INTERNAL_H */
