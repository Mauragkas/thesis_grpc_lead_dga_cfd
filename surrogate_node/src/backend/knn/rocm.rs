use super::ffi::{knn_model_create, knn_model_free, knn_predict_rocm, KnnModelHandle};
use super::traits::KnnBackend;
use crate::backend::common::{BackendError, GpDeviceType};
use crate::data::scaler::StandardScaler;

/// ROCm/HIP k-Nearest Neighbours backend stub.
/// Currently falls back to the CPU OpenMP implementation in `knn_rocm.cpp`.
pub struct KnnRocmBackend {
    handle: *mut KnnModelHandle,
    n_train: usize,
    dim: usize,
    k: u32,
}

unsafe impl Send for KnnRocmBackend {}
unsafe impl Sync for KnnRocmBackend {}

impl KnnRocmBackend {
    pub fn new(
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        dim: usize,
        length_scales: &[f64],
        k: u32,
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
        Ok(Self { handle, n_train, dim, k })
    }
}

impl Drop for KnnRocmBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { knn_model_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl KnnBackend for KnnRocmBackend {
    fn device_type(&self) -> GpDeviceType { GpDeviceType::Rocm }
    fn device_name(&self) -> &str { "AMD ROCm (k-NN)" }
    fn k(&self) -> u32 { self.k }
    fn n_train(&self) -> usize { self.n_train }
    fn dim(&self) -> usize { self.dim }

    fn predict(&self, x_test: &[f64], n_test: usize) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        let mut pred = vec![0.0f64; n_test];
        let mut dist = vec![0.0f64; n_test];
        let status = unsafe {
            knn_predict_rocm(
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

pub fn create_knn_rocm(
    x_train_raw: &[f64],
    y_train: &[f64],
    n_train: usize,
    dim: usize,
    k: u32,
) -> Result<(KnnRocmBackend, StandardScaler), BackendError> {
    let scaler = StandardScaler::fit(x_train_raw, n_train, dim);
    let x_scaled = scaler.transform(x_train_raw);
    let length_scales = vec![1.0f64; dim];
    let backend = KnnRocmBackend::new(&x_scaled, y_train, n_train, dim, &length_scales, k)?;
    Ok((backend, scaler))
}
