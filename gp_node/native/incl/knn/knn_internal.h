#ifndef KNN_INTERNAL_H
#define KNN_INTERNAL_H

#include "knn_backend.h"
#include <vector>
#include <cstddef>
#include <cstdint>

struct KnnModelHandle {
    std::vector<double> X_train;       /* n_train * dim, row-major C-order */
    std::vector<double> y_train;       /* n_train                          */
    std::vector<double> length_scales; /* dim                              */
    size_t              n_train;
    size_t              dim;
    uint32_t            k;
};

#endif /* KNN_INTERNAL_H */
