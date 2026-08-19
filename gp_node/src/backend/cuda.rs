use super::ffi::*;
use super::traits::ComputeBackend;
use crate::domain::{GpHyperparameters, KernelType};
use std::sync::atomic::{AtomicPtr, Ordering};

pub struct CudaBackend {
    handle: AtomicPtr<GpBackendHandle>,
    name: String,
    device_id: i32,
}

impl CudaBackend {
    pub fn new(device_id: i32) -> Result<Self, BackendError> {
        let mut handle_ptr: *mut GpBackendHandle = std::ptr::null_mut();
        let status = unsafe { gp_backend_create(GpDeviceType::Cuda, device_id, &mut handle_ptr) };
        status.to_result()?;
        if handle_ptr.is_null() {
            return Err(BackendError::NullPointer);
        }

        Ok(Self {
            handle: AtomicPtr::new(handle_ptr),
            name: format!("NVIDIA CUDA GPU #{}", device_id),
            device_id,
        })
    }

    fn get_raw_handle(&self) -> *mut GpBackendHandle {
        self.handle.load(Ordering::Relaxed)
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        let handle = self.handle.swap(std::ptr::null_mut(), Ordering::SeqCst);
        if !handle.is_null() {
            unsafe { gp_backend_free(handle) };
        }
    }
}

impl ComputeBackend for CudaBackend {
    fn device_type(&self) -> GpDeviceType {
        GpDeviceType::Cuda
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn compute_covariance(
        &self,
        x1: &[f64],
        n1: usize,
        x2: &[f64],
        n2: usize,
        dim: usize,
        params: &GpHyperparameters,
        add_diagonal_noise: bool,
    ) -> Result<Vec<f64>, BackendError> {
        assert_eq!(x1.len(), n1 * dim);
        assert_eq!(x2.len(), n2 * dim);
        assert_eq!(params.length_scales.len(), dim);

        let ktype = match params.kernel_type {
            KernelType::Matern52 => GpKernelType::Matern52,
            KernelType::Rbf => GpKernelType::Rbf,
        };

        let ffi_params = GpHyperparamsFFI {
            length_scales: params.length_scales.as_ptr(),
            num_dim: dim as u32,
            signal_variance: params.signal_variance,
            noise_variance: params.noise_variance,
            kernel_type: ktype,
        };

        let mut out_k = vec![0.0; n1 * n2];
        let status = unsafe {
            gp_compute_covariance(
                self.get_raw_handle(),
                x1.as_ptr(),
                n1,
                x2.as_ptr(),
                n2,
                dim,
                &ffi_params,
                add_diagonal_noise,
                out_k.as_mut_ptr(),
            )
        };

        status.to_result()?;
        Ok(out_k)
    }

    fn cholesky(&self, k: &[f64], n: usize) -> Result<Vec<f64>, BackendError> {
        assert_eq!(k.len(), n * n);
        let mut out_l = vec![0.0; n * n];

        let status = unsafe {
            gp_cholesky(
                self.get_raw_handle(),
                k.as_ptr(),
                n,
                out_l.as_mut_ptr(),
            )
        };

        status.to_result()?;
        Ok(out_l)
    }

    fn compute_alpha(&self, l: &[f64], y: &[f64], n: usize) -> Result<Vec<f64>, BackendError> {
        assert_eq!(l.len(), n * n);
        assert_eq!(y.len(), n);

        let mut out_alpha = vec![0.0; n];
        let status = unsafe {
            gp_compute_alpha(
                self.get_raw_handle(),
                l.as_ptr(),
                y.as_ptr(),
                n,
                out_alpha.as_mut_ptr(),
            )
        };

        status.to_result()?;
        Ok(out_alpha)
    }

    fn predict_batch(
        &self,
        x_train: &[f64],
        n_train: usize,
        x_test: &[f64],
        n_test: usize,
        dim: usize,
        alpha: &[f64],
        l: &[f64],
        params: &GpHyperparameters,
    ) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        let ktype = match params.kernel_type {
            KernelType::Matern52 => GpKernelType::Matern52,
            KernelType::Rbf => GpKernelType::Rbf,
        };

        let ffi_params = GpHyperparamsFFI {
            length_scales: params.length_scales.as_ptr(),
            num_dim: dim as u32,
            signal_variance: params.signal_variance,
            noise_variance: params.noise_variance,
            kernel_type: ktype,
        };

        let mut out_mean = vec![0.0; n_test];
        let mut out_var = vec![0.0; n_test];

        let status = unsafe {
            gp_predict_batch(
                self.get_raw_handle(),
                x_train.as_ptr(),
                n_train,
                x_test.as_ptr(),
                n_test,
                dim,
                alpha.as_ptr(),
                l.as_ptr(),
                &ffi_params,
                out_mean.as_mut_ptr(),
                out_var.as_mut_ptr(),
            )
        };

        status.to_result()?;
        Ok((out_mean, out_var))
    }

    fn log_marginal_likelihood(
        &self,
        y: &[f64],
        alpha: &[f64],
        l: &[f64],
        n: usize,
    ) -> Result<f64, BackendError> {
        let mut out_mll = 0.0;
        let status = unsafe {
            gp_log_marginal_likelihood(
                self.get_raw_handle(),
                y.as_ptr(),
                alpha.as_ptr(),
                l.as_ptr(),
                n,
                &mut out_mll,
            )
        };

        status.to_result()?;
        Ok(out_mll)
    }
}
