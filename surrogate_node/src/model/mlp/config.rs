use crate::backend::mlp::ffi::{MlpActivationFFI, MlpHyperparamsFFI};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpActivation {
    Relu,
    Silu,
    Gelu,
    Tanh,
}

impl MlpActivation {
    pub fn to_ffi(self) -> MlpActivationFFI {
        match self {
            MlpActivation::Relu => MlpActivationFFI::Relu,
            MlpActivation::Silu => MlpActivationFFI::Silu,
            MlpActivation::Gelu => MlpActivationFFI::Gelu,
            MlpActivation::Tanh => MlpActivationFFI::Tanh,
        }
    }
}

/// Hyperparameter configuration for Multi-Layer Perceptron (Neural Network).
#[derive(Debug, Clone)]
pub struct MlpConfig {
    pub hidden_layers: Vec<usize>,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub activation: MlpActivation,
    pub seed: u64,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![64, 32],
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            batch_size: 32,
            epochs: 300,
            activation: MlpActivation::Silu,
            seed: 42,
        }
    }
}

impl MlpConfig {
    pub fn to_ffi(&self) -> MlpHyperparamsFFI {
        let mut hidden = [0u32; 4];
        let n = self.hidden_layers.len().min(4);
        for (slot, &val) in hidden[..n].iter_mut().zip(&self.hidden_layers[..n]) {
            *slot = val as u32;
        }

        MlpHyperparamsFFI {
            hidden_layers: hidden,
            num_hidden_layers: n as u32,
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            batch_size: self.batch_size as u32,
            epochs: self.epochs as u32,
            activation: self.activation.to_ffi(),
            seed: self.seed,
        }
    }
}
