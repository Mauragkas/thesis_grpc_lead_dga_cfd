use crate::backend::common::BackendError;
use crate::domain::EvaluationMetrics;

/// High-level surrogate model interface for regression and evaluation.
pub trait SurrogateModel {
    /// Name of the compute device used by this surrogate instance.
    fn device_name(&self) -> &str;

    /// Evaluates model predictions on raw test features `x_test_raw` and ground truth `y_test`.
    fn evaluate(&self, x_test_raw: &[f64], y_test: &[f64]) -> Result<EvaluationMetrics, BackendError>;
}
