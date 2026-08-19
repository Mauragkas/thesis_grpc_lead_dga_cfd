use super::cpu::KnnCpuBackend;
use super::cuda::KnnCudaBackend;
use super::rocm::KnnRocmBackend;
use super::traits::KnnBackend;
use crate::backend::common::{get_best_available_device, BackendError, GpDeviceType};

pub struct KnnBackendFactory;

impl KnnBackendFactory {
    /// Detects the best available hardware and returns a fitted k-NN backend
    /// on scaled data.
    pub fn create_best(
        x_train_scaled: &[f64],
        y_train_scaled: &[f64],
        n_train: usize,
        dim: usize,
        k: u32,
    ) -> Box<dyn KnnBackend> {
        let best_dev = get_best_available_device();
        let ls = vec![1.0f64; dim];

        match best_dev {
            GpDeviceType::Cuda => {
                if let Ok(b) = KnnCudaBackend::new(x_train_scaled, y_train_scaled, n_train, dim, &ls, k, 0) {
                    return Box::new(b);
                }
            }
            GpDeviceType::Rocm => {
                if let Ok(b) = KnnRocmBackend::new(x_train_scaled, y_train_scaled, n_train, dim, &ls, k) {
                    return Box::new(b);
                }
            }
            GpDeviceType::Cpu => {}
        }

        let b = KnnCpuBackend::new(x_train_scaled, y_train_scaled, n_train, dim, &ls, k)
            .expect("Failed to create CPU k-NN backend");
        Box::new(b)
    }

    /// Explicitly creates a k-NN backend on the requested device.
    pub fn create(
        device_type: GpDeviceType,
        x_train_scaled: &[f64],
        y_train_scaled: &[f64],
        n_train: usize,
        dim: usize,
        k: u32,
    ) -> Result<Box<dyn KnnBackend>, BackendError> {
        let ls = vec![1.0f64; dim];

        let backend: Box<dyn KnnBackend> = match device_type {
            GpDeviceType::Cpu => Box::new(KnnCpuBackend::new(x_train_scaled, y_train_scaled, n_train, dim, &ls, k)?),
            GpDeviceType::Cuda => Box::new(KnnCudaBackend::new(x_train_scaled, y_train_scaled, n_train, dim, &ls, k, 0)?),
            GpDeviceType::Rocm => Box::new(KnnRocmBackend::new(x_train_scaled, y_train_scaled, n_train, dim, &ls, k)?),
        };
        Ok(backend)
    }
}
