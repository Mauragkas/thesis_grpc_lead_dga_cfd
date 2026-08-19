use crate::backend::common::{BackendError, GpDeviceType};
use crate::domain::GpHyperparameters;

/// Abstract hardware compute backend for Gaussian Process linear algebra & kernel operations.
pub trait ComputeBackend: Send + Sync {
    /// Returns the device type (CPU, CUDA, ROCm).
    fn device_type(&self) -> GpDeviceType;

    /// Returns human-readable device name.
    fn device_name(&self) -> &str;

    /// Computes covariance matrix K between X1 (N1 x dim) and X2 (N2 x dim).
    #[allow(clippy::too_many_arguments)]
    fn compute_covariance(
        &self,
        x1: &[f64],
        n1: usize,
        x2: &[f64],
        n2: usize,
        dim: usize,
        params: &GpHyperparameters,
        add_diagonal_noise: bool,
    ) -> Result<Vec<f64>, BackendError>;

    /// Computes lower Cholesky decomposition L of positive-definite matrix K (N x N).
    fn cholesky(&self, k: &[f64], n: usize) -> Result<Vec<f64>, BackendError>;

    /// Computes alpha weights vector: alpha = (K + sigma_n^2 * I)^(-1) * y.
    fn compute_alpha(&self, l: &[f64], y: &[f64], n: usize) -> Result<Vec<f64>, BackendError>;

    /// Evaluates predicted mean and variance for test batch X_test.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(Vec<f64>, Vec<f64>), BackendError>;

    /// Evaluates marginal log-likelihood (MLL).
    fn log_marginal_likelihood(
        &self,
        y: &[f64],
        alpha: &[f64],
        l: &[f64],
        n: usize,
    ) -> Result<f64, BackendError>;
}
