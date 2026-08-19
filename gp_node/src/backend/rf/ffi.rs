use crate::backend::common::BackendError;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RfStatusCode {
    Success = 0,
    ErrorNullPointer = -1,
    ErrorDimensionMismatch = -2,
    ErrorInvalidParam = -3,
    ErrorCudaFail = -4,
    ErrorRocmFail = -5,
    ErrorAllocFail = -6,
    ErrorUnknown = -99,
}

impl RfStatusCode {
    pub fn to_result(self) -> Result<(), BackendError> {
        match self {
            RfStatusCode::Success => Ok(()),
            RfStatusCode::ErrorNullPointer => Err(BackendError::NullPointer),
            RfStatusCode::ErrorDimensionMismatch => Err(BackendError::DimensionMismatch),
            RfStatusCode::ErrorInvalidParam => Err(BackendError::NumericalError),
            RfStatusCode::ErrorCudaFail => Err(BackendError::CudaFailure),
            RfStatusCode::ErrorRocmFail => Err(BackendError::RocmFailure),
            RfStatusCode::ErrorAllocFail => Err(BackendError::Unknown(-6)),
            RfStatusCode::ErrorUnknown => Err(BackendError::Unknown(-99)),
        }
    }
}

/// Hyperparameters struct mirroring `RfHyperparams` in rf_backend.h.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RfHyperparamsFFI {
    pub n_estimators: u32,
    pub max_depth: u32,
    pub max_features: u32,
    pub min_samples_leaf: u32,
    pub seed: u64,
}

/// Opaque handle for the RF forest owned by the C++ layer.
#[repr(C)]
pub struct RfModelHandle {
    _private: [u8; 0],
}

extern "C" {
    pub fn rf_model_train(
        x_train: *const f64,
        y_train: *const f64,
        n_train: usize,
        dim: usize,
        params: *const RfHyperparamsFFI,
        out_handle: *mut *mut RfModelHandle,
    ) -> RfStatusCode;

    pub fn rf_model_free(handle: *mut RfModelHandle);

    pub fn rf_predict(
        handle: *const RfModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
        out_tree_preds: *mut f64,
    ) -> RfStatusCode;

    pub fn rf_predict_cuda(
        handle: *const RfModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
        out_tree_preds: *mut f64,
    ) -> RfStatusCode;

    pub fn rf_predict_rocm(
        handle: *const RfModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
        out_tree_preds: *mut f64,
    ) -> RfStatusCode;

    pub fn rf_get_n_estimators(handle: *const RfModelHandle) -> u32;
    pub fn rf_get_max_depth(handle: *const RfModelHandle) -> u32;
    pub fn rf_get_n_nodes_total(handle: *const RfModelHandle) -> usize;
}
