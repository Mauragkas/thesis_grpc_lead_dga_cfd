use super::cpu::CpuOpenMpBackend;
use super::cuda::CudaBackend;
use super::ffi::*;
use super::rocm::RocmBackend;
use super::traits::ComputeBackend;
use std::ffi::CStr;

pub struct BackendFactory;

impl BackendFactory {
    /// Probes system for available devices and returns device descriptors.
    pub fn probe_devices() -> Vec<String> {
        let mut devices = [GpDeviceInfoFFI {
            device_type: GpDeviceType::Cpu,
            name: [0; 128],
            device_id: 0,
            num_compute_units: 0,
            total_memory_bytes: 0,
        }; 16];

        let mut count = 0;
        let status = unsafe { gp_probe_devices(devices.as_mut_ptr(), 16, &mut count) };
        if status != GpStatusCode::Success || count <= 0 {
            return vec!["Host CPU (Fallback)".to_string()];
        }

        let mut names = Vec::new();
        for dev in devices.iter().take(count as usize) {
            let c_str = unsafe { CStr::from_ptr(dev.name.as_ptr()) };
            names.push(c_str.to_string_lossy().into_owned());
        }
        names
    }

    /// Automatically detects hardware capabilities and creates the highest performance backend.
    pub fn create_best_backend() -> Box<dyn ComputeBackend> {
        let best_dev = unsafe { gp_get_best_device() };
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
