use crate::backend::common::BackendError;
use crate::data::splitter::DatasetSplitter;
use crate::domain::EvaluationMetrics;
use crate::model::mlp::config::MlpConfig;
use crate::model::mlp::surrogate::MlpSurrogate;
use tracing::info;

/// Single Responsibility: trains an `MlpSurrogate` on a given dataset
/// and evaluates holdout validation metrics.
pub struct SurrogateTrainer;

impl SurrogateTrainer {
    pub fn train_mlp(
        x_data: &[f64],
        y_data: &[f64],
        n_samples: usize,
        dim: usize,
        config: MlpConfig,
    ) -> Result<(MlpSurrogate, EvaluationMetrics), BackendError> {
        assert_eq!(x_data.len(), n_samples * dim);
        assert_eq!(y_data.len(), n_samples);

        if n_samples < 5 {
            return Err(BackendError::DimensionMismatch);
        }

        // If we have at least 15 samples, split train/val/test using split_60_20_20
        if n_samples >= 15 {
            let split = DatasetSplitter::split_60_20_20(x_data, y_data, n_samples, dim, 42);

            let surrogate = MlpSurrogate::fit(
                &split.train.x,
                &split.train.y,
                split.train.num_samples,
                split.train.dim,
                config,
            )?;

            let metrics = surrogate.evaluate(&split.validation.x, &split.validation.y)?;
            info!(
                "Trained MLP model on {} samples (val {} samples): R2={:.4}, RMSE={:.4}, MAE={:.4}",
                split.train.num_samples,
                split.validation.num_samples,
                metrics.r2_score,
                metrics.rmse,
                metrics.mae
            );
            Ok((surrogate, metrics))
        } else {
            // Train on all samples and evaluate on the same set
            let surrogate = MlpSurrogate::fit(x_data, y_data, n_samples, dim, config)?;
            let metrics = surrogate.evaluate(x_data, y_data)?;
            info!(
                "Trained MLP model on all {} samples (no split): R2={:.4}, RMSE={:.4}",
                n_samples, metrics.r2_score, metrics.rmse
            );
            Ok((surrogate, metrics))
        }
    }
}
