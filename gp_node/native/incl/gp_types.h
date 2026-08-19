#ifndef GP_TYPES_H
#define GP_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GP_SUCCESS = 0,
    GP_ERROR_NULL_POINTER = -1,
    GP_ERROR_DIMENSION_MISMATCH = -2,
    GP_ERROR_NOT_POSITIVE_DEFINITE = -3,
    GP_ERROR_CUDA_FAIL = -4,
    GP_ERROR_ROCM_FAIL = -5,
    GP_ERROR_UNSUPPORTED_DEVICE = -6,
    GP_ERROR_NUMERICAL = -7,
    GP_ERROR_UNKNOWN = -99
} GpStatusCode;

typedef enum {
    GP_KERNEL_MATERN52 = 0,
    GP_KERNEL_RBF = 1
} GpKernelType;

typedef enum {
    GP_DEVICE_CPU = 0,
    GP_DEVICE_CUDA = 1,
    GP_DEVICE_ROCM = 2
} GpDeviceType;

typedef struct {
    const double* length_scales;  /* Array of length scales per dimension [dim] */
    uint32_t num_dim;             /* Dimensionality D */
    double signal_variance;       /* sigma_f^2 */
    double noise_variance;        /* sigma_n^2 */
    GpKernelType kernel_type;     /* Matérn 5/2 or RBF */
} GpHyperparams;

typedef struct {
    GpDeviceType device_type;
    char name[128];
    int device_id;
    int num_compute_units;
    size_t total_memory_bytes;
} GpDeviceInfo;

#ifdef __cplusplus
}
#endif

#endif /* GP_TYPES_H */
