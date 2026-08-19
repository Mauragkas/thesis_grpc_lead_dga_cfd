use crate::backend::common::{BackendError, GpDeviceType};

/// Compute backend for k-Nearest Neighbours regression.
pub trait KnnBackend: Send + Sync {
    /// Returns the hardware device type in use.
    fn device_type(&self) -> GpDeviceType;

    /// Human-readable device name (for logging and reporting).
    fn device_name(&self) -> &str;

    /// Value of k used for nearest-neighbour retrieval.
    fn k(&self) -> u32;

    /// Number of training samples stored in the model.
    fn n_train(&self) -> usize;

    /// Feature dimensionality.
    fn dim(&self) -> usize;

    /// Predicts targets for `n_test` query points in scaled feature space.
    fn predict(
        &self,
        x_test: &[f64],
        n_test: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), BackendError>;
}
