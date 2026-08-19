use crate::backend::common::BackendError;
use crate::backend::knn::factory::KnnBackendFactory;
use crate::backend::knn::traits::KnnBackend;
use crate::data::scaler::{StandardScaler, TargetScaler};
use crate::domain::EvaluationMetrics;
use crate::model::traits::SurrogateModel;

/// High-level k-Nearest Neighbours surrogate.
///
/// Wraps a hardware backend, owns the scalers fitted on training data,
/// and manages transformations transparently.
pub struct KnnSurrogate {
    backend: Box<dyn KnnBackend>,
    x_scaler: StandardScaler,
    y_scaler: TargetScaler,
    dim: usize,
    n_train: usize,
}

impl KnnSurrogate {
    pub fn from_backend(
        backend: Box<dyn KnnBackend>,
        x_scaler: StandardScaler,
        y_scaler: TargetScaler,
        dim: usize,
        n_train: usize,
    ) -> Self {
        Self { backend, x_scaler, y_scaler, dim, n_train }
    }

    /// Fits the k-NN surrogate: standardises X and y, selects best hardware backend.
    pub fn fit(
        x_train_raw: &[f64],
        y_train_raw: &[f64],
        n_train: usize,
        dim: usize,
        k: u32,
    ) -> Result<Self, BackendError> {
        let x_scaler = StandardScaler::fit(x_train_raw, n_train, dim);
        let y_scaler = TargetScaler::fit(y_train_raw);

        let x_scaled = x_scaler.transform(x_train_raw);
        let y_scaled = y_scaler.transform(y_train_raw);

        let backend = KnnBackendFactory::create_best(&x_scaled, &y_scaled, n_train, dim, k);
        Ok(Self { backend, x_scaler, y_scaler, dim, n_train })
    }

    pub fn device_name(&self) -> &str { self.backend.device_name() }
    pub fn k(&self) -> u32 { self.backend.k() }
    pub fn n_train(&self) -> usize { self.n_train }
    pub fn dim(&self) -> usize { self.dim }

    /// Predicts fitness values for raw (unscaled) test points.
    /// Returns `(predictions, mean_knn_distances)`.
    pub fn predict(&self, x_test_raw: &[f64]) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        let n_test = x_test_raw.len() / self.dim;
        let x_scaled = self.x_scaler.transform(x_test_raw);

        let (pred_scaled, dist) = self.backend.predict(&x_scaled, n_test)?;

        // Inverse-transform predictions back to original target space
        let pred_raw = self.y_scaler.inverse_transform(&pred_scaled);
        Ok((pred_raw, dist))
    }

    /// Evaluates the surrogate on a raw holdout set.
    pub fn evaluate(&self, x_test_raw: &[f64], y_test: &[f64]) -> Result<EvaluationMetrics, BackendError> {
        let (y_pred, _dist) = self.predict(x_test_raw)?;
        Ok(EvaluationMetrics::compute(y_test, &y_pred))
    }
}

impl SurrogateModel for KnnSurrogate {
    fn device_name(&self) -> &str {
        self.device_name()
    }

    fn evaluate(&self, x_test_raw: &[f64], y_test: &[f64]) -> Result<EvaluationMetrics, BackendError> {
        self.evaluate(x_test_raw, y_test)
    }
}
