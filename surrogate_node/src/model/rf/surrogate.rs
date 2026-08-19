use super::config::RfConfig;
use crate::backend::common::BackendError;
use crate::backend::rf::factory::RfBackendFactory;
use crate::backend::rf::traits::RfBackend;
use crate::data::scaler::{StandardScaler, TargetScaler};
use crate::domain::EvaluationMetrics;
use crate::model::traits::SurrogateModel;

/// High-level Random Forest regression surrogate.
///
/// Delegates compute to a hardware backend (`RfCpuBackend`, `RfCudaBackend`, or
/// `RfRocmBackend`) and manages feature / target normalisation.
pub struct RfSurrogate {
    backend: Box<dyn RfBackend>,
    x_scaler: StandardScaler,
    y_scaler: TargetScaler,
    config: RfConfig,
    dim: usize,
    n_train: usize,
}

impl RfSurrogate {
    pub fn from_backend(
        backend: Box<dyn RfBackend>,
        x_scaler: StandardScaler,
        y_scaler: TargetScaler,
        config: RfConfig,
        dim: usize,
        n_train: usize,
    ) -> Self {
        Self { backend, x_scaler, y_scaler, config, dim, n_train }
    }

    /// Fits the surrogate: scales features + targets, trains the forest.
    pub fn fit(
        x_train_raw: &[f64],
        y_train_raw: &[f64],
        n_train: usize,
        dim: usize,
        config: RfConfig,
    ) -> Result<Self, BackendError> {
        let x_scaler = StandardScaler::fit(x_train_raw, n_train, dim);
        let y_scaler = TargetScaler::fit(y_train_raw);

        let x_scaled = x_scaler.transform(x_train_raw);
        let y_scaled = y_scaler.transform(y_train_raw);

        let backend = RfBackendFactory::create_best(&x_scaled, &y_scaled, n_train, dim, config.to_ffi());
        Ok(Self { backend, x_scaler, y_scaler, config, dim, n_train })
    }

    pub fn device_name(&self) -> &str { self.backend.device_name() }
    pub fn config(&self) -> &RfConfig { &self.config }
    pub fn n_train(&self) -> usize { self.n_train }
    pub fn dim(&self) -> usize { self.dim }

    /// Predicts mean targets for raw (unscaled) query points.
    pub fn predict(&self, x_test_raw: &[f64]) -> Result<Vec<f64>, BackendError> {
        let n_test = x_test_raw.len() / self.dim;
        let x_scaled = self.x_scaler.transform(x_test_raw);
        let pred_scaled = self.backend.predict(&x_scaled, n_test)?;
        Ok(self.y_scaler.inverse_transform(&pred_scaled))
    }

    /// Predicts mean and standard deviation for raw query points.
    /// The std is derived from inter-tree variance, analogous to GP posterior std.
    pub fn predict_with_std(&self, x_test_raw: &[f64]) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        let n_test = x_test_raw.len() / self.dim;
        let x_scaled = self.x_scaler.transform(x_test_raw);
        let (pred_scaled, var_scaled) = self.backend.predict_with_variance(&x_scaled, n_test)?;

        let pred_raw = self.y_scaler.inverse_transform(&pred_scaled);
        // Scale variance back to original target units
        let std_scale = self.y_scaler.std;
        let std_raw: Vec<f64> = var_scaled
            .iter()
            .map(|&v| (v * std_scale * std_scale).max(0.0).sqrt())
            .collect();
        Ok((pred_raw, std_raw))
    }

    /// Evaluates on a raw holdout set.
    pub fn evaluate(&self, x_test_raw: &[f64], y_test: &[f64]) -> Result<EvaluationMetrics, BackendError> {
        let y_pred = self.predict(x_test_raw)?;
        Ok(EvaluationMetrics::compute(y_test, &y_pred))
    }
}

impl SurrogateModel for RfSurrogate {
    fn device_name(&self) -> &str {
        self.device_name()
    }

    fn evaluate(&self, x_test_raw: &[f64], y_test: &[f64]) -> Result<EvaluationMetrics, BackendError> {
        self.evaluate(x_test_raw, y_test)
    }
}
