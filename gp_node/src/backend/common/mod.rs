pub mod device;
pub mod error;

pub use device::{
    get_best_available_device, probe_available_devices, GpDeviceInfo, GpDeviceInfoFFI, GpDeviceType,
};
pub use error::{BackendError, GpStatusCode};
