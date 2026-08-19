use super::cpu::CpuOpenMpBackend;
use super::cuda::CudaBackend;
use super::rocm::RocmBackend;
use super::traits::ComputeBackend;
use crate::backend::common::{
    get_best_available_device, probe_available_devices, BackendError, GpDeviceType,
};

pub struct BackendFactory;

impl BackendFactory {
    /// Probes system for available devices and returns device descriptor strings.
    pub fn probe_devices() -> Vec<String> {
        probe_available_devices()
            .into_iter()
            .map(|d| d.to_string())
            .collect()
    }

    /// Automatically detects hardware capabilities and creates the highest performance backend.
    pub fn create_best_backend() -> Box<dyn ComputeBackend> {
        let best_dev = get_best_available_device();
        match best_dev {
            GpDeviceType::Cuda => {
                if let Ok(b) = CudaBackend::new(0) {
                    return Box::new(b);
                }
            }
            GpDeviceType::Rocm => {
                if let Ok(b) = RocmBackend::new(0) {
                    return Box::new(b);
                }
            }
            GpDeviceType::Cpu => {}
        }

        // Fallback to CPU OpenMP backend
        Box::new(CpuOpenMpBackend::new().expect("Failed to initialize CPU OpenMP backend"))
    }

    /// Explicitly creates a backend for a requested device type and index.
    pub fn create_backend(
        device_type: GpDeviceType,
        device_id: i32,
    ) -> Result<Box<dyn ComputeBackend>, BackendError> {
        match device_type {
            GpDeviceType::Cpu => Ok(Box::new(CpuOpenMpBackend::new()?)),
            GpDeviceType::Cuda => Ok(Box::new(CudaBackend::new(device_id)?)),
            GpDeviceType::Rocm => Ok(Box::new(RocmBackend::new(device_id)?)),
        }
    }
}
