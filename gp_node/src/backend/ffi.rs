use std::os::raw::c_char;
use thiserror::Error;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpStatusCode {
    Success = 0,
    ErrorNullPointer = -1,
    ErrorDimensionMismatch = -2,
    ErrorNotPositiveDefinite = -3,
    ErrorCudaFail = -4,
    ErrorRocmFail = -5,
    ErrorUnsupportedDevice = -6,
    ErrorNumerical = -7,
    ErrorUnknown = -99,
}

impl GpStatusCode {
    pub fn to_result(self) -> Result<(), BackendError> {
        match self {
            GpStatusCode::Success => Ok(()),
            GpStatusCode::ErrorNullPointer => Err(BackendError::NullPointer),
            GpStatusCode::ErrorDimensionMismatch => Err(BackendError::DimensionMismatch),
            GpStatusCode::ErrorNotPositiveDefinite => Err(BackendError::NotPositiveDefinite),
            GpStatusCode::ErrorCudaFail => Err(BackendError::CudaFailure),
            GpStatusCode::ErrorRocmFail => Err(BackendError::RocmFailure),
            GpStatusCode::ErrorUnsupportedDevice => Err(BackendError::UnsupportedDevice),
            GpStatusCode::ErrorNumerical => Err(BackendError::NumericalError),
            GpStatusCode::ErrorUnknown => Err(BackendError::Unknown(-99)),
        }
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Null pointer passed to native backend")]
    NullPointer,
    #[error("Dimension mismatch in matrix computation")]
    DimensionMismatch,
    #[error("Covariance matrix is not positive-definite (Cholesky failed)")]
    NotPositiveDefinite,
    #[error("CUDA GPU kernel execution failed")]
    CudaFailure,
    #[error("ROCm GPU kernel execution failed")]
    RocmFailure,
    #[error("Unsupported device")]
    UnsupportedDevice,
    #[error("Numerical instability in solver (e.g. division by zero or NaN)")]
    NumericalError,
    #[error("Unknown backend error (code: {0})")]
    Unknown(i32),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpKernelType {
    Matern52 = 0,
    Rbf = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpDeviceType {
    Cpu = 0,
    Cuda = 1,
    Rocm = 2,
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
#[derive(Clone, Copy)]
pub struct GpDeviceInfoFFI {
    pub device_type: GpDeviceType,
    pub name: [c_char; 128],
    pub device_id: i32,
    pub num_compute_units: i32,
    pub total_memory_bytes: usize,
}

#[repr(C)]
pub struct GpBackendHandle {
    _private: [u8; 0],
}

extern "C" {
    pub fn gp_probe_devices(
        out_devices: *mut GpDeviceInfoFFI,
        max_devices: i32,
        out_count: *mut i32,
    ) -> GpStatusCode;

    pub fn gp_get_best_device() -> GpDeviceType;

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
