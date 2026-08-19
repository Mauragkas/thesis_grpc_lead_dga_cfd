#include "gp_device.h"
#include "gp_cuda.h"
#include "gp_rocm.h"
#include <cstring>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

GpStatusCode gp_probe_devices(GpDeviceInfo* out_devices, int max_devices, int* out_count) {
    if (!out_devices || max_devices <= 0 || !out_count) {
        return GP_ERROR_NULL_POINTER;
    }

    int count = 0;

    /* 1. Probe CUDA Devices */
    int cuda_count = 0;
    if (cuda_is_available(&cuda_count) && cuda_count > 0) {
        for (int i = 0; i < cuda_count && count < max_devices; ++i) {
            GpDeviceInfo& info = out_devices[count++];
            info.device_type = GP_DEVICE_CUDA;
            info.device_id = i;
            std::snprintf(info.name, sizeof(info.name), "NVIDIA CUDA GPU #%d", i);
            info.num_compute_units = 0;
            info.total_memory_bytes = 0;
        }
    }

    /* 2. Probe ROCm Devices */
    int rocm_count = 0;
    if (rocm_is_available(&rocm_count) && rocm_count > 0) {
        for (int i = 0; i < rocm_count && count < max_devices; ++i) {
            GpDeviceInfo& info = out_devices[count++];
            info.device_type = GP_DEVICE_ROCM;
            info.device_id = i;
            std::snprintf(info.name, sizeof(info.name), "AMD ROCm GPU #%d", i);
            info.num_compute_units = 0;
            info.total_memory_bytes = 0;
        }
    }

    /* 3. Probe Host CPU */
    if (count < max_devices) {
        GpDeviceInfo& info = out_devices[count++];
        info.device_type = GP_DEVICE_CPU;
        info.device_id = 0;
        int threads = 1;
#ifdef _OPENMP
        threads = omp_get_max_threads();
#endif
        std::snprintf(info.name, sizeof(info.name), "Host CPU (OpenMP %d threads)", threads);
        info.num_compute_units = threads;
        info.total_memory_bytes = 0;
    }

    *out_count = count;
    return GP_SUCCESS;
}

GpDeviceType gp_get_best_device(void) {
    int cuda_count = 0;
    if (cuda_is_available(&cuda_count) && cuda_count > 0) {
        return GP_DEVICE_CUDA;
    }

    int rocm_count = 0;
    if (rocm_is_available(&rocm_count) && rocm_count > 0) {
        return GP_DEVICE_ROCM;
    }

    return GP_DEVICE_CPU;
}
