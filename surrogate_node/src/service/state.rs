use crate::domain::EvaluationMetrics;
use crate::model::mlp::surrogate::MlpSurrogate;
use crate::service::buffer::SlidingWindowBuffer;
use crate::service::config::SurrogateConfig;
use crate::service::trainer::SurrogateTrainer;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info};

/// Thread-safe shared state for the surrogate microservice.
/// Coordinates sliding window ingestion, zero-downtime model prediction,
/// and periodic background training.
pub struct SurrogateState {
    pub config: SurrogateConfig,
    buffer: Mutex<SlidingWindowBuffer>,
    active_model: RwLock<Option<Arc<MlpSurrogate>>>,
    model_version: AtomicU64,
    samples_since_retrain: AtomicUsize,
    is_training: AtomicBool,
    last_metrics: RwLock<Option<EvaluationMetrics>>,
}

impl SurrogateState {
    pub fn new(config: SurrogateConfig) -> Arc<Self> {
        let capacity = config.window_size;
        Arc::new(Self {
            config,
            buffer: Mutex::new(SlidingWindowBuffer::new(capacity)),
            active_model: RwLock::new(None),
            model_version: AtomicU64::new(0),
            samples_since_retrain: AtomicUsize::new(0),
            is_training: AtomicBool::new(false),
            last_metrics: RwLock::new(None),
        })
    }

    /// Ingests a batch of true evaluation samples. If sample count interval is met,
    /// triggers background retraining asynchronously.
    pub async fn ingest_samples(
        self: &Arc<Self>,
        samples: Vec<(Vec<f64>, f64)>,
        force_train: bool,
    ) -> (usize, bool) {
        let (window_len, total_ingested) = {
            let mut buf = self.buffer.lock().await;
            let count = buf.push_batch(samples);
            (buf.len(), count)
        };

        let current_since = self.samples_since_retrain.fetch_add(total_ingested, Ordering::SeqCst) + total_ingested;
        let should_train = force_train || (current_since >= self.config.retrain_interval);

        let mut training_started = false;
        if should_train && window_len >= self.config.min_train_samples {
            training_started = self.trigger_train_async(force_train).await;
        }

        (window_len, training_started)
    }

    /// Triggers asynchronous training on a snapshot of the current sliding window buffer.
    pub async fn trigger_train_async(self: &Arc<Self>, _force: bool) -> bool {
        // Ensure only one training run executes at a time
        if self
            .is_training
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            info!("Training already in progress; skipping trigger");
            return false;
        }

        let dataset = {
            let buf = self.buffer.lock().await;
            buf.extract_dataset()
        };

        let (x_data, y_data, n_samples, dim) = match dataset {
            Some(d) if d.2 >= self.config.min_train_samples => d,
            _ => {
                self.is_training.store(false, Ordering::SeqCst);
                return false;
            }
        };

        let this = self.clone();
        let mlp_cfg = self.config.to_mlp_config(42 + self.model_version.load(Ordering::Relaxed));

        tokio::task::spawn_blocking(move || {
            info!(
                "Starting background MLP retraining on {} samples (dim={})...",
                n_samples, dim
            );
            match SurrogateTrainer::train_mlp(&x_data, &y_data, n_samples, dim, mlp_cfg) {
                Ok((surrogate, metrics)) => {
                    let version = this.model_version.fetch_add(1, Ordering::SeqCst) + 1;
                    info!(
                        "Background retraining complete! Model v{} active (R2={:.4}, RMSE={:.4})",
                        version, metrics.r2_score, metrics.rmse
                    );

                    // Atomic zero-downtime swap of the active model
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async {
                        let mut active = this.active_model.write().await;
                        *active = Some(Arc::new(surrogate));
                        let mut m = this.last_metrics.write().await;
                        *m = Some(metrics);
                    });

                    this.samples_since_retrain.store(0, Ordering::SeqCst);
                }
                Err(e) => {
                    error!("Background MLP retraining failed: {e}");
                }
            }
            this.is_training.store(false, Ordering::SeqCst);
        });

        true
    }

    /// Evaluates a batch of individuals using the active model.
    pub async fn predict_batch(&self, individuals: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let guard = self.active_model.read().await;
        let model = match guard.as_ref() {
            Some(m) => m.clone(),
            None => return Err("Model is not ready (still in cold-start)".to_string()),
        };

        if individuals.is_empty() {
            return Ok(Vec::new());
        }

        let dim = model.dim();
        let mut x_flat = Vec::with_capacity(individuals.len() * dim);
        for ind in individuals {
            if ind.len() != dim {
                return Err(format!(
                    "Dimension mismatch: expected {dim}, got {}",
                    ind.len()
                ));
            }
            x_flat.extend_from_slice(ind);
        }

        model.predict(&x_flat).map_err(|e| format!("Prediction error: {e}"))
    }

    /// Checks if the surrogate is initialized and ready to serve predictions.
    pub async fn is_ready(&self) -> bool {
        self.active_model.read().await.is_some()
    }

    /// Returns the current model version.
    pub fn model_version(&self) -> u64 {
        self.model_version.load(Ordering::Relaxed)
    }

    /// Returns the current sliding window buffer length.
    pub async fn window_size(&self) -> usize {
        self.buffer.lock().await.len()
    }

    /// Returns current device name if model is active.
    pub async fn device_name(&self) -> String {
        let guard = self.active_model.read().await;
        guard
            .as_ref()
            .map(|m| m.device_name().to_string())
            .unwrap_or_else(|| "uninitialized".to_string())
    }

    /// Returns last validation metrics if available.
    pub async fn last_metrics(&self) -> Option<EvaluationMetrics> {
        self.last_metrics.read().await.clone()
    }
}
