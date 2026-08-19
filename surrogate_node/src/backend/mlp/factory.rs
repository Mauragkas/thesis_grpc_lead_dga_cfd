use super::cpu::MlpCpuBackend;
use super::cuda::MlpCudaBackend;
use super::ffi::MlpHyperparamsFFI;
use super::rocm::MlpRocmBackend;
use super::traits::MlpBackend;
use crate::backend::common::{get_best_available_device, BackendError, GpDeviceType};

pub struct MlpBackendFactory;

impl MlpBackendFactory {
    /// Detects best hardware and returns a fitted Multi-Layer Perceptron backend.
    pub fn create_best(
        x_train_scaled: &[f64],
        y_train_scaled: &[f64],
        n_train: usize,
        dim: usize,
        params: MlpHyperparamsFFI,
    ) -> Box<dyn MlpBackend> {
        let best_dev = get_best_available_device();

        match best_dev {
            GpDeviceType::Cuda => {
                if let Ok(b) = MlpCudaBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params) {
                    return Box::new(b);
                }
            }
            GpDeviceType::Rocm => {
                if let Ok(b) = MlpRocmBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params) {
                    return Box::new(b);
                }
            }
            GpDeviceType::Cpu => {}
        }

        let b = MlpCpuBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params)
            .expect("Failed to create CPU MLP backend");
        Box::new(b)
    }

    /// Explicitly creates a Multi-Layer Perceptron backend on the requested device.
    pub fn create(
        device_type: GpDeviceType,
        x_train_scaled: &[f64],
        y_train_scaled: &[f64],
        n_train: usize,
        dim: usize,
        params: MlpHyperparamsFFI,
    ) -> Result<Box<dyn MlpBackend>, BackendError> {
        let backend: Box<dyn MlpBackend> = match device_type {
            GpDeviceType::Cpu => Box::new(MlpCpuBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params)?),
            GpDeviceType::Cuda => Box::new(MlpCudaBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params)?),
            GpDeviceType::Rocm => Box::new(MlpRocmBackend::new(x_train_scaled, y_train_scaled, n_train, dim, params)?),
        };
        Ok(backend)
    }
}
