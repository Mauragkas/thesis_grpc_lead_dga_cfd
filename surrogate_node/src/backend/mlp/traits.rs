use crate::backend::common::{BackendError, GpDeviceType};

/// Compute backend for Multi-Layer Perceptron (Neural Network) regression.
pub trait MlpBackend: Send + Sync {
    /// Returns the hardware device type used for inference.
    fn device_type(&self) -> GpDeviceType;

    /// Human-readable device name.
    fn device_name(&self) -> &str;

    /// Number of training samples.
    fn n_train(&self) -> usize;

    /// Feature dimensionality.
    fn dim(&self) -> usize;

    /// Total number of trainable weights and biases in the neural network.
    fn total_parameters(&self) -> usize;

    /// Predicts mean targets for `n_test` query points.
    fn predict(&self, x_test: &[f64], n_test: usize) -> Result<Vec<f64>, BackendError>;
}
