use std::ffi::CStr;
use std::os::raw::c_char;
use super::error::GpStatusCode;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpDeviceType {
    Cpu = 0,
    Cuda = 1,
    Rocm = 2,
}

impl std::fmt::Display for GpDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpDeviceType::Cpu => write!(f, "CPU (OpenMP)"),
            GpDeviceType::Cuda => write!(f, "CUDA GPU"),
            GpDeviceType::Rocm => write!(f, "ROCm GPU"),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpDeviceInfoFFI {
    pub device_type: GpDeviceType,
    pub name: [c_char; 128],
    pub device_id: i32,
    pub num_compute_units: i32,
    pub total_memory_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct GpDeviceInfo {
    pub device_type: GpDeviceType,
    pub name: String,
    pub device_id: i32,
    pub num_compute_units: i32,
    pub total_memory_bytes: usize,
}

impl std::fmt::Display for GpDeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [ID: {}] ({} compute units, {:.1} MB VRAM)",
            self.name,
            self.device_id,
            self.num_compute_units,
            self.total_memory_bytes as f64 / (1024.0 * 1024.0)
        )
    }
}

extern "C" {
    pub fn gp_probe_devices(
        out_devices: *mut GpDeviceInfoFFI,
        max_devices: i32,
        out_count: *mut i32,
    ) -> GpStatusCode;

    pub fn gp_get_best_device() -> GpDeviceType;
}

pub fn probe_available_devices() -> Vec<GpDeviceInfo> {
    let mut raw_devices = [GpDeviceInfoFFI {
        device_type: GpDeviceType::Cpu,
        name: [0; 128],
        device_id: -1,
        num_compute_units: 0,
        total_memory_bytes: 0,
    }; 16];
    let mut count: i32 = 0;

    let status = unsafe { gp_probe_devices(raw_devices.as_mut_ptr(), 16, &mut count) };
    if status != GpStatusCode::Success || count <= 0 {
        return vec![GpDeviceInfo {
            device_type: GpDeviceType::Cpu,
            name: "Generic OpenMP CPU (Fallback)".to_string(),
            device_id: 0,
            num_compute_units: 1,
            total_memory_bytes: 0,
        }];
    }

    let mut out = Vec::with_capacity(count as usize);
    for dev in raw_devices.iter().take(count as usize) {
        let name_str = unsafe { CStr::from_ptr(dev.name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        out.push(GpDeviceInfo {
            device_type: dev.device_type,
            name: name_str,
            device_id: dev.device_id,
            num_compute_units: dev.num_compute_units,
            total_memory_bytes: dev.total_memory_bytes,
        });
    }
    out
}

pub fn get_best_available_device() -> GpDeviceType {
    unsafe { gp_get_best_device() }
}
