use super::cpu::RfCpuBackend;
use super::cuda::RfCudaBackend;
use super::ffi::RfHyperparamsFFI;
use super::rocm::RfRocmBackend;
use super::traits::RfBackend;
use crate::backend::common::{get_best_available_device, BackendError, GpDeviceType};

pub struct RfBackendFactory;

impl RfBackendFactory {
    /// Detects best hardware and returns a fitted Random Forest backend.
    /// Training always executes on CPU; GPU is used only for inference.
    pub fn create_best(
        x_train_scaled: &[f64],
        y_train_scaled: &[f64],
        n_train: usize,
        dim: usize,
        params: RfHyperparamsFFI,
    ) -> Box<dyn RfBackend> {
        let best_dev = get_best_available_device();

        match best_dev {
            GpDeviceType::Cuda => {
                if let Ok(b) = RfCudaBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params) {
                    return Box::new(b);
                }
            }
            GpDeviceType::Rocm => {
                if let Ok(b) = RfRocmBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params) {
                    return Box::new(b);
                }
            }
            GpDeviceType::Cpu => {}
        }

        let b = RfCpuBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params)
            .expect("Failed to create CPU Random Forest backend");
        Box::new(b)
    }

    /// Explicitly creates a Random Forest backend on the requested device.
    pub fn create(
        device_type: GpDeviceType,
        x_train_scaled: &[f64],
        y_train_scaled: &[f64],
        n_train: usize,
        dim: usize,
        params: RfHyperparamsFFI,
    ) -> Result<Box<dyn RfBackend>, BackendError> {
        let backend: Box<dyn RfBackend> = match device_type {
            GpDeviceType::Cpu => Box::new(RfCpuBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params)?),
            GpDeviceType::Cuda => Box::new(RfCudaBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params)?),
            GpDeviceType::Rocm => Box::new(RfRocmBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params)?),
        };
        Ok(backend)
    }
}
