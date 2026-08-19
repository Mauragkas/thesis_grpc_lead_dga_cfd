#ifndef GP_DEVICE_H
#define GP_DEVICE_H

#include "gp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Probes the system for available compute devices (CUDA GPUs, ROCm GPUs, and CPU).
 * @param out_devices Array to receive device info structs.
 * @param max_devices Maximum capacity of out_devices array.
 * @param out_count Pointer to receive the actual number of detected devices.
 * @return GP_SUCCESS on success.
 */
GpStatusCode gp_probe_devices(GpDeviceInfo* out_devices, int max_devices, int* out_count);

/**
 * Returns the recommended best device available on the host system.
 */
GpDeviceType gp_get_best_device(void);

#ifdef __cplusplus
}
#endif

#endif /* GP_DEVICE_H */
