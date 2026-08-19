use super::ffi::{knn_model_create, knn_model_free, knn_predict_cuda, KnnModelHandle};
use super::traits::KnnBackend;
use crate::backend::common::{BackendError, GpDeviceType};
use crate::data::scaler::StandardScaler;

/// CUDA k-Nearest Neighbours backend.
/// Training data stored on host; inference dispatched to GPU via `knn_cuda.cu`.
pub struct KnnCudaBackend {
    handle: *mut KnnModelHandle,
    n_train: usize,
    dim: usize,
    k: u32,
    _device_id: i32,
}

unsafe impl Send for KnnCudaBackend {}
unsafe impl Sync for KnnCudaBackend {}

impl KnnCudaBackend {
    pub fn new(
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        dim: usize,
        length_scales: &[f64],
        k: u32,
        device_id: i32,
    ) -> Result<Self, BackendError> {
        let mut handle: *mut KnnModelHandle = std::ptr::null_mut();
        let status = unsafe {
            knn_model_create(
                x_train.as_ptr(),
                y_train.as_ptr(),
                n_train,
                dim,
                length_scales.as_ptr(),
                k,
                &mut handle,
            )
        };
        status.to_result()?;
        Ok(Self { handle, n_train, dim, k, _device_id: device_id })
    }
}

impl Drop for KnnCudaBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { knn_model_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl KnnBackend for KnnCudaBackend {
    fn device_type(&self) -> GpDeviceType { GpDeviceType::Cuda }
    fn device_name(&self) -> &str { "NVIDIA CUDA (k-NN)" }
    fn k(&self) -> u32 { self.k }
    fn n_train(&self) -> usize { self.n_train }
    fn dim(&self) -> usize { self.dim }

    fn predict(&self, x_test: &[f64], n_test: usize) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        let mut pred = vec![0.0f64; n_test];
        let mut dist = vec![0.0f64; n_test];
        let status = unsafe {
            knn_predict_cuda(
                self.handle,
                x_test.as_ptr(),
                n_test,
                pred.as_mut_ptr(),
                dist.as_mut_ptr(),
            )
        };
        status.to_result()?;
        Ok((pred, dist))
    }
}

pub fn create_knn_cuda(
    x_train_raw: &[f64],
    y_train: &[f64],
    n_train: usize,
    dim: usize,
    k: u32,
    device_id: i32,
) -> Result<(KnnCudaBackend, StandardScaler), BackendError> {
    let scaler = StandardScaler::fit(x_train_raw, n_train, dim);
    let x_scaled = scaler.transform(x_train_raw);
    let length_scales = vec![1.0f64; dim];
    let backend = KnnCudaBackend::new(&x_scaled, y_train, n_train, dim, &length_scales, k, device_id)?;
    Ok((backend, scaler))
}
