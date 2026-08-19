use super::ffi::{rf_model_free, rf_model_train, rf_predict_cuda, RfHyperparamsFFI, RfModelHandle};
use super::traits::RfBackend;
use crate::backend::common::{BackendError, GpDeviceType};

/// CUDA Random Forest backend: training on CPU OpenMP, inference on GPU.
pub struct RfCudaBackend {
    handle: *mut RfModelHandle,
    n_train: usize,
    dim: usize,
    params: RfHyperparamsFFI,
}

unsafe impl Send for RfCudaBackend {}
unsafe impl Sync for RfCudaBackend {}

impl RfCudaBackend {
    pub fn new(
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        dim: usize,
        params: RfHyperparamsFFI,
    ) -> Result<Self, BackendError> {
        let mut handle: *mut RfModelHandle = std::ptr::null_mut();
        let status = unsafe {
            rf_model_train(
                x_train.as_ptr(),
                y_train.as_ptr(),
                n_train,
                dim,
                &params as *const RfHyperparamsFFI,
                &mut handle,
            )
        };
        status.to_result()?;
        Ok(Self { handle, n_train, dim, params })
    }
}

impl Drop for RfCudaBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { rf_model_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl RfBackend for RfCudaBackend {
    fn device_type(&self) -> GpDeviceType { GpDeviceType::Cuda }
    fn device_name(&self) -> &str { "NVIDIA CUDA (Random Forest inference)" }
    fn n_estimators(&self) -> u32 { self.params.n_estimators }
    fn max_depth(&self) -> u32 { self.params.max_depth }
    fn n_train(&self) -> usize { self.n_train }
    fn dim(&self) -> usize { self.dim }

    fn predict(&self, x_test: &[f64], n_test: usize) -> Result<Vec<f64>, BackendError> {
        let mut pred = vec![0.0f64; n_test];
        let status = unsafe {
            rf_predict_cuda(
                self.handle,
                x_test.as_ptr(),
                n_test,
                pred.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        status.to_result()?;
        Ok(pred)
    }

    fn predict_with_variance(
        &self,
        x_test: &[f64],
        n_test: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        let n_trees = self.params.n_estimators as usize;
        let mut pred = vec![0.0f64; n_test];
        let mut tree_preds = vec![0.0f64; n_trees * n_test];
        let status = unsafe {
            rf_predict_cuda(
                self.handle,
                x_test.as_ptr(),
                n_test,
                pred.as_mut_ptr(),
                tree_preds.as_mut_ptr(),
            )
        };
        status.to_result()?;
        let mut variance = vec![0.0f64; n_test];
        for m in 0..n_test {
            let mean = pred[m];
            let sq_sum: f64 = (0..n_trees).map(|t| { let tp = tree_preds[t * n_test + m]; tp * tp }).sum();
            variance[m] = (sq_sum / n_trees as f64 - mean * mean).max(0.0);
        }
        Ok((pred, variance))
    }
}
