use crate::backend::common::{GpDeviceType, GpStatusCode};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpKernelType {
    Matern52 = 0,
    Rbf = 1,
}

#[repr(C)]
pub struct GpHyperparamsFFI {
    pub length_scales: *const f64,
    pub num_dim: u32,
    pub signal_variance: f64,
    pub noise_variance: f64,
    pub kernel_type: GpKernelType,
}

#[repr(C)]
pub struct GpBackendHandle {
    _private: [u8; 0],
}

extern "C" {
    pub fn gp_backend_create(
        device_type: GpDeviceType,
        device_id: i32,
        out_handle: *mut *mut GpBackendHandle,
    ) -> GpStatusCode;

    pub fn gp_backend_free(handle: *mut GpBackendHandle);

    pub fn gp_backend_get_type(handle: *const GpBackendHandle) -> GpDeviceType;

    pub fn gp_compute_covariance(
        handle: *mut GpBackendHandle,
        x1: *const f64,
        n1: usize,
        x2: *const f64,
        n2: usize,
        dim: usize,
        params: *const GpHyperparamsFFI,
        add_diagonal_noise: bool,
        out_k: *mut f64,
    ) -> GpStatusCode;

    pub fn gp_cholesky(
        handle: *mut GpBackendHandle,
        k: *const f64,
        n: usize,
        out_l: *mut f64,
    ) -> GpStatusCode;

    pub fn gp_compute_alpha(
        handle: *mut GpBackendHandle,
        l: *const f64,
        y: *const f64,
        n: usize,
        out_alpha: *mut f64,
    ) -> GpStatusCode;

    pub fn gp_predict_batch(
        handle: *mut GpBackendHandle,
        x_train: *const f64,
        n_train: usize,
        x_test: *const f64,
        n_test: usize,
        dim: usize,
        alpha: *const f64,
        l: *const f64,
        params: *const GpHyperparamsFFI,
        out_mean: *mut f64,
        out_var: *mut f64,
    ) -> GpStatusCode;

    pub fn gp_log_marginal_likelihood(
        handle: *mut GpBackendHandle,
        y: *const f64,
        alpha: *const f64,
        l: *const f64,
        n: usize,
        out_mll: *mut f64,
    ) -> GpStatusCode;
}
