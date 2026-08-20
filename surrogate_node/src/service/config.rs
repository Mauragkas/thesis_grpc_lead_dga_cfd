use crate::model::mlp::config::{MlpActivation, MlpConfig};
use std::env;
use tracing::info;

/// Single Responsibility: encapsulates and parses runtime configuration
/// for the surrogate gRPC service and its online training pipeline.
#[derive(Debug, Clone)]
pub struct SurrogateConfig {
    pub grpc_bind: String,
    pub window_size: usize,
    pub retrain_interval: usize,
    pub min_train_samples: usize,
    pub mlp_epochs: usize,
    pub mlp_lr: f64,
    pub mlp_batch_size: usize,
    pub mlp_hidden_layers: Vec<usize>,
    pub mlp_activation: MlpActivation,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            grpc_bind: "0.0.0.0:50054".to_string(),
            window_size: 1000,
            retrain_interval: 20,
            min_train_samples: 30,
            mlp_epochs: 250,
            mlp_lr: 1e-3,
            mlp_batch_size: 32,
            mlp_hidden_layers: vec![64, 32],
            mlp_activation: MlpActivation::Silu,
        }
    }
}

impl SurrogateConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(val) = env::var("GRPC_BIND").or_else(|_| env::var("SURROGATE_GRPC_BIND")) {
            cfg.grpc_bind = val;
        }
        if let Ok(val) = env::var("WINDOW_SIZE").or_else(|_| env::var("SURROGATE_WINDOW_SIZE")) {
            if let Ok(n) = val.parse::<usize>() {
                cfg.window_size = n;
            }
        }
        if let Ok(val) = env::var("RETRAIN_INTERVAL").or_else(|_| env::var("SURROGATE_RETRAIN_INTERVAL")) {
            if let Ok(n) = val.parse::<usize>() {
                cfg.retrain_interval = n;
            }
        }
        if let Ok(val) = env::var("MIN_TRAIN_SAMPLES").or_else(|_| env::var("SURROGATE_MIN_TRAIN_SAMPLES")) {
            if let Ok(n) = val.parse::<usize>() {
                cfg.min_train_samples = n;
            }
        }
        if let Ok(val) = env::var("MLP_EPOCHS").or_else(|_| env::var("SURROGATE_MLP_EPOCHS")) {
            if let Ok(n) = val.parse::<usize>() {
                cfg.mlp_epochs = n;
            }
        }
        if let Ok(val) = env::var("MLP_LR").or_else(|_| env::var("SURROGATE_MLP_LR")) {
            if let Ok(lr) = val.parse::<f64>() {
                cfg.mlp_lr = lr;
            }
        }
        if let Ok(val) = env::var("MLP_BATCH_SIZE").or_else(|_| env::var("SURROGATE_MLP_BATCH_SIZE")) {
            if let Ok(bs) = val.parse::<usize>() {
                cfg.mlp_batch_size = bs;
            }
        }
        if let Ok(val) = env::var("MLP_HIDDEN").or_else(|_| env::var("SURROGATE_MLP_HIDDEN")) {
            let layers: Vec<usize> = val
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            if !layers.is_empty() {
                cfg.mlp_hidden_layers = layers;
            }
        }

        info!(
            "SurrogateConfig: bind={}, window_size={}, retrain_interval={}, min_samples={}, epochs={}, lr={}, hidden={:?}",
            cfg.grpc_bind,
            cfg.window_size,
            cfg.retrain_interval,
            cfg.min_train_samples,
            cfg.mlp_epochs,
            cfg.mlp_lr,
            cfg.mlp_hidden_layers
        );

        cfg
    }

    pub fn to_mlp_config(&self, seed: u64) -> MlpConfig {
        MlpConfig {
            hidden_layers: self.mlp_hidden_layers.clone(),
            learning_rate: self.mlp_lr,
            weight_decay: 1e-4,
            batch_size: self.mlp_batch_size,
            epochs: self.mlp_epochs,
            activation: self.mlp_activation,
            seed,
        }
    }
}
