use crate::backend::common::BackendError;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnStatusCode {
    Success = 0,
    ErrorNullPointer = -1,
    ErrorDimensionMismatch = -2,
    ErrorInvalidK = -3,
    ErrorCudaFail = -4,
    ErrorRocmFail = -5,
    ErrorAllocFail = -6,
    ErrorUnknown = -99,
}

impl KnnStatusCode {
    pub fn to_result(self) -> Result<(), BackendError> {
        match self {
            KnnStatusCode::Success => Ok(()),
            KnnStatusCode::ErrorNullPointer => Err(BackendError::NullPointer),
            KnnStatusCode::ErrorDimensionMismatch => Err(BackendError::DimensionMismatch),
            KnnStatusCode::ErrorInvalidK => Err(BackendError::NumericalError),
            KnnStatusCode::ErrorCudaFail => Err(BackendError::CudaFailure),
            KnnStatusCode::ErrorRocmFail => Err(BackendError::RocmFailure),
            KnnStatusCode::ErrorAllocFail => Err(BackendError::Unknown(-6)),
            KnnStatusCode::ErrorUnknown => Err(BackendError::Unknown(-99)),
        }
    }
}

/// Opaque handle for the k-NN model owned by the C++ layer.
#[repr(C)]
pub struct KnnModelHandle {
    _private: [u8; 0],
}

extern "C" {
    pub fn knn_model_create(
        x_train: *const f64,
        y_train: *const f64,
        n_train: usize,
        dim: usize,
        length_scales: *const f64,
        k: u32,
        out_handle: *mut *mut KnnModelHandle,
    ) -> KnnStatusCode;

    pub fn knn_model_free(handle: *mut KnnModelHandle);

    pub fn knn_predict(
        handle: *const KnnModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
        out_dist: *mut f64,
    ) -> KnnStatusCode;

    pub fn knn_predict_cuda(
        handle: *const KnnModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
        out_dist: *mut f64,
    ) -> KnnStatusCode;

    pub fn knn_predict_rocm(
        handle: *const KnnModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
        out_dist: *mut f64,
    ) -> KnnStatusCode;
}
