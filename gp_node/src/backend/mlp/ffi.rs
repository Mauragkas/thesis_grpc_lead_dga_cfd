use crate::backend::common::BackendError;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpStatusCode {
    Success = 0,
    ErrorNullPointer = -1,
    ErrorDimensionMismatch = -2,
    ErrorInvalidParam = -3,
    ErrorCudaFail = -4,
    ErrorRocmFail = -5,
    ErrorAllocFail = -6,
    ErrorUnknown = -99,
}

impl MlpStatusCode {
    pub fn to_result(self) -> Result<(), BackendError> {
        match self {
            MlpStatusCode::Success => Ok(()),
            MlpStatusCode::ErrorNullPointer => Err(BackendError::NullPointer),
            MlpStatusCode::ErrorDimensionMismatch => Err(BackendError::DimensionMismatch),
            MlpStatusCode::ErrorInvalidParam => Err(BackendError::NumericalError),
            MlpStatusCode::ErrorCudaFail => Err(BackendError::CudaFailure),
            MlpStatusCode::ErrorRocmFail => Err(BackendError::RocmFailure),
            MlpStatusCode::ErrorAllocFail => Err(BackendError::Unknown(-6)),
            MlpStatusCode::ErrorUnknown => Err(BackendError::Unknown(-99)),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpActivationFFI {
    Relu = 0,
    Silu = 1,
    Gelu = 2,
    Tanh = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlpHyperparamsFFI {
    pub hidden_layers: [u32; 4],
    pub num_hidden_layers: u32,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub batch_size: u32,
    pub epochs: u32,
    pub activation: MlpActivationFFI,
    pub seed: u64,
}

/// Opaque handle for the MLP neural network model owned by C++ layer.
#[repr(C)]
pub struct MlpModelHandle {
    _private: [u8; 0],
}

extern "C" {
    pub fn mlp_model_train(
        x_train: *const f64,
        y_train: *const f64,
        n_train: usize,
        dim: usize,
        params: *const MlpHyperparamsFFI,
        out_handle: *mut *mut MlpModelHandle,
    ) -> MlpStatusCode;

    pub fn mlp_model_free(handle: *mut MlpModelHandle);

    pub fn mlp_predict(
        handle: *const MlpModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
    ) -> MlpStatusCode;

    pub fn mlp_predict_cuda(
        handle: *const MlpModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
    ) -> MlpStatusCode;

    pub fn mlp_predict_rocm(
        handle: *const MlpModelHandle,
        x_test: *const f64,
        n_test: usize,
        out_pred: *mut f64,
    ) -> MlpStatusCode;

    pub fn mlp_get_num_layers(handle: *const MlpModelHandle) -> u32;
    pub fn mlp_get_total_parameters(handle: *const MlpModelHandle) -> usize;
}
