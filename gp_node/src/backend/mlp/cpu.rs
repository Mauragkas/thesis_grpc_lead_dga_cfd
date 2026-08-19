use super::ffi::{
    mlp_get_total_parameters, mlp_model_free, mlp_model_train, mlp_predict, MlpHyperparamsFFI,
    MlpModelHandle,
};
use super::traits::MlpBackend;
use crate::backend::common::{BackendError, GpDeviceType};

/// CPU OpenMP Multi-Layer Perceptron backend.
pub struct MlpCpuBackend {
    handle: *mut MlpModelHandle,
    n_train: usize,
    dim: usize,
}

unsafe impl Send for MlpCpuBackend {}
unsafe impl Sync for MlpCpuBackend {}

impl MlpCpuBackend {
    pub fn new(
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        dim: usize,
        params: MlpHyperparamsFFI,
    ) -> Result<Self, BackendError> {
        assert_eq!(x_train.len(), n_train * dim);
        assert_eq!(y_train.len(), n_train);

        let mut handle: *mut MlpModelHandle = std::ptr::null_mut();
        let status = unsafe {
            mlp_model_train(
                x_train.as_ptr(),
                y_train.as_ptr(),
                n_train,
                dim,
                &params as *const MlpHyperparamsFFI,
                &mut handle,
            )
        };
        status.to_result()?;
        Ok(Self { handle, n_train, dim })
    }
}

impl Drop for MlpCpuBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { mlp_model_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl MlpBackend for MlpCpuBackend {
    fn device_type(&self) -> GpDeviceType { GpDeviceType::Cpu }
    fn device_name(&self) -> &str { "CPU OpenMP (Neural Network MLP)" }
    fn n_train(&self) -> usize { self.n_train }
    fn dim(&self) -> usize { self.dim }
    fn total_parameters(&self) -> usize {
        unsafe { mlp_get_total_parameters(self.handle) }
    }

    fn predict(&self, x_test: &[f64], n_test: usize) -> Result<Vec<f64>, BackendError> {
        assert_eq!(x_test.len(), n_test * self.dim);
        let mut pred = vec![0.0f64; n_test];
        let status = unsafe {
            mlp_predict(
                self.handle,
                x_test.as_ptr(),
                n_test,
                pred.as_mut_ptr(),
            )
        };
        status.to_result()?;
        Ok(pred)
    }
}
