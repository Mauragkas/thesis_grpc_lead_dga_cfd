use super::config::MlpConfig;
use crate::backend::common::BackendError;
use crate::backend::mlp::factory::MlpBackendFactory;
use crate::backend::mlp::traits::MlpBackend;
use crate::data::scaler::{StandardScaler, TargetScaler};
use crate::domain::EvaluationMetrics;
use crate::model::traits::SurrogateModel;

/// High-level Multi-Layer Perceptron (Neural Network) regression surrogate.
pub struct MlpSurrogate {
    backend: Box<dyn MlpBackend>,
    x_scaler: StandardScaler,
    y_scaler: TargetScaler,
    config: MlpConfig,
    dim: usize,
    n_train: usize,
}

impl MlpSurrogate {
    pub fn from_backend(
        backend: Box<dyn MlpBackend>,
        x_scaler: StandardScaler,
        y_scaler: TargetScaler,
        config: MlpConfig,
        dim: usize,
        n_train: usize,
    ) -> Self {
        Self { backend, x_scaler, y_scaler, config, dim, n_train }
    }

    /// Fits the neural network surrogate: standardises X and y, trains the MLP.
    pub fn fit(
        x_train_raw: &[f64],
        y_train_raw: &[f64],
        n_train: usize,
        dim: usize,
        config: MlpConfig,
    ) -> Result<Self, BackendError> {
        let x_scaler = StandardScaler::fit(x_train_raw, n_train, dim);
        let y_scaler = TargetScaler::fit(y_train_raw);

        let x_scaled = x_scaler.transform(x_train_raw);
        let y_scaled = y_scaler.transform(y_train_raw);

        let backend = MlpBackendFactory::create_best(&x_scaled, &y_scaled, n_train, dim, config.to_ffi());
        Ok(Self { backend, x_scaler, y_scaler, config, dim, n_train })
    }

    pub fn device_name(&self) -> &str { self.backend.device_name() }
    pub fn config(&self) -> &MlpConfig { &self.config }
    pub fn n_train(&self) -> usize { self.n_train }
    pub fn dim(&self) -> usize { self.dim }
    pub fn total_parameters(&self) -> usize { self.backend.total_parameters() }

    /// Predicts target values for raw (unscaled) query points.
    pub fn predict(&self, x_test_raw: &[f64]) -> Result<Vec<f64>, BackendError> {
        let n_test = x_test_raw.len() / self.dim;
        let x_scaled = self.x_scaler.transform(x_test_raw);
        let pred_scaled = self.backend.predict(&x_scaled, n_test)?;
        Ok(self.y_scaler.inverse_transform(&pred_scaled))
    }

    /// Evaluates the neural network surrogate on a raw holdout set.
    pub fn evaluate(&self, x_test_raw: &[f64], y_test: &[f64]) -> Result<EvaluationMetrics, BackendError> {
        let y_pred = self.predict(x_test_raw)?;
        Ok(EvaluationMetrics::compute(y_test, &y_pred))
    }
}

impl SurrogateModel for MlpSurrogate {
    fn device_name(&self) -> &str {
        self.device_name()
    }

    fn evaluate(&self, x_test_raw: &[f64], y_test: &[f64]) -> Result<EvaluationMetrics, BackendError> {
        self.evaluate(x_test_raw, y_test)
    }
}
