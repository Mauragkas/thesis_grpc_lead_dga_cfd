use super::ffi::{rf_model_free, rf_model_train, rf_predict, RfHyperparamsFFI, RfModelHandle};
use super::traits::RfBackend;
use crate::backend::common::{BackendError, GpDeviceType};

/// CPU OpenMP Random Forest backend.
/// Wraps `rf_cpu.cpp` via FFI; training is fully parallel (one thread per tree).
pub struct RfCpuBackend {
    handle: *mut RfModelHandle,
    n_train: usize,
    dim: usize,
    params: RfHyperparamsFFI,
}

unsafe impl Send for RfCpuBackend {}
unsafe impl Sync for RfCpuBackend {}

impl RfCpuBackend {
    pub fn new(
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        dim: usize,
        params: RfHyperparamsFFI,
    ) -> Result<Self, BackendError> {
        assert_eq!(x_train.len(), n_train * dim);
        assert_eq!(y_train.len(), n_train);

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

impl Drop for RfCpuBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { rf_model_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl RfBackend for RfCpuBackend {
    fn device_type(&self) -> GpDeviceType { GpDeviceType::Cpu }
    fn device_name(&self) -> &str { "CPU OpenMP (Random Forest)" }
    fn n_estimators(&self) -> u32 { self.params.n_estimators }
    fn max_depth(&self) -> u32 { self.params.max_depth }
    fn n_train(&self) -> usize { self.n_train }
    fn dim(&self) -> usize { self.dim }

    fn predict(&self, x_test: &[f64], n_test: usize) -> Result<Vec<f64>, BackendError> {
        assert_eq!(x_test.len(), n_test * self.dim);
        let mut pred = vec![0.0f64; n_test];
        let status = unsafe {
            rf_predict(
                self.handle,
                x_test.as_ptr(),
                n_test,
                pred.as_mut_ptr(),
                std::ptr::null_mut(), // tree_preds not needed here
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
            rf_predict(
                self.handle,
                x_test.as_ptr(),
                n_test,
                pred.as_mut_ptr(),
                tree_preds.as_mut_ptr(),
            )
        };
        status.to_result()?;

        // Variance across trees: Var = E[t^2] - E[t]^2
        let mut variance = vec![0.0f64; n_test];
        for m in 0..n_test {
            let mean = pred[m];
            let mut sq_sum = 0.0;
            for t in 0..n_trees {
                let tp = tree_preds[t * n_test + m];
                sq_sum += tp * tp;
            }
            variance[m] = (sq_sum / n_trees as f64 - mean * mean).max(0.0);
        }
        Ok((pred, variance))
    }
}
