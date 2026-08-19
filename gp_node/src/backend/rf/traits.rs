use crate::backend::common::{BackendError, GpDeviceType};

/// Compute backend for Random Forest regression.
pub trait RfBackend: Send + Sync {
    /// Returns the hardware device type used for **inference**.
    fn device_type(&self) -> GpDeviceType;

    /// Human-readable device name.
    fn device_name(&self) -> &str;

    /// Number of trees in the ensemble.
    fn n_estimators(&self) -> u32;

    /// Maximum depth of each tree.
    fn max_depth(&self) -> u32;

    /// Number of training samples.
    fn n_train(&self) -> usize;

    /// Feature dimensionality.
    fn dim(&self) -> usize;

    /// Predicts mean targets for `n_test` query points.
    fn predict(&self, x_test: &[f64], n_test: usize) -> Result<Vec<f64>, BackendError>;

    /// Predicts mean and inter-tree variance (empirical uncertainty) for query points.
    fn predict_with_variance(
        &self,
        x_test: &[f64],
        n_test: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), BackendError>;
}
