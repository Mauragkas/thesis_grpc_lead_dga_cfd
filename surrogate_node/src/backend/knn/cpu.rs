use super::ffi::{knn_model_create, knn_model_free, knn_predict, KnnModelHandle};
use super::traits::KnnBackend;
use crate::backend::common::{BackendError, GpDeviceType};
use crate::data::scaler::StandardScaler;

/// CPU OpenMP k-Nearest Neighbours backend.
/// Wraps the C++ `knn_cpu.cpp` implementation via FFI.
pub struct KnnCpuBackend {
    handle: *mut KnnModelHandle,
    n_train: usize,
    dim: usize,
    k: u32,
}

// SAFETY: The underlying C++ handle is thread-safe for read-only predict calls.
unsafe impl Send for KnnCpuBackend {}
unsafe impl Sync for KnnCpuBackend {}

impl KnnCpuBackend {
    /// Creates a new CPU k-NN backend and fits it on the training data.
    pub fn new(
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        dim: usize,
        length_scales: &[f64],
        k: u32,
    ) -> Result<Self, BackendError> {
        assert_eq!(x_train.len(), n_train * dim);
        assert_eq!(y_train.len(), n_train);
        assert_eq!(length_scales.len(), dim);
        assert!(k > 0, "k must be at least 1");

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

impl Drop for KnnCpuBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { knn_model_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl KnnBackend for KnnCpuBackend {
    fn device_type(&self) -> GpDeviceType { GpDeviceType::Cpu }
    fn device_name(&self) -> &str { "CPU OpenMP (k-NN)" }
    fn k(&self) -> u32 { self.k }
    fn n_train(&self) -> usize { self.n_train }
    fn dim(&self) -> usize { self.dim }

    fn predict(&self, x_test: &[f64], n_test: usize) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        assert_eq!(x_test.len(), n_test * self.dim);
        let mut pred = vec![0.0f64; n_test];
        let mut dist = vec![0.0f64; n_test];
        let status = unsafe {
            knn_predict(
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

/// Convenience constructor: fits StandardScaler, creates handle, returns backend + scaler.
pub fn create_knn_cpu(
    x_train_raw: &[f64],
    y_train: &[f64],
    n_train: usize,
    dim: usize,
    k: u32,
) -> Result<(KnnCpuBackend, StandardScaler), BackendError> {
    let scaler = StandardScaler::fit(x_train_raw, n_train, dim);
    let x_scaled = scaler.transform(x_train_raw);
    let length_scales = vec![1.0f64; dim];
    let backend = KnnCpuBackend::new(&x_scaled, y_train, n_train, dim, &length_scales, k)?;
    Ok((backend, scaler))
}
