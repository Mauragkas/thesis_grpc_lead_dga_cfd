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
