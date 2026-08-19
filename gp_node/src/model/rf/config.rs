use crate::backend::rf::ffi::RfHyperparamsFFI;

/// Hyperparameter set for the Random Forest surrogate.
#[derive(Debug, Clone, Copy)]
pub struct RfConfig {
    pub n_estimators: u32,
    pub max_depth: u32,
    pub max_features: u32,
    pub min_samples_leaf: u32,
    pub seed: u64,
}

impl Default for RfConfig {
    fn default() -> Self {
        Self {
            n_estimators: 150,
            max_depth: 12,
            max_features: 3, // ≈ sqrt(11) for our 11-feature dataset
            min_samples_leaf: 1,
            seed: 42,
        }
    }
}

impl RfConfig {
    pub fn to_ffi(&self) -> RfHyperparamsFFI {
        RfHyperparamsFFI {
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            max_features: self.max_features,
            min_samples_leaf: self.min_samples_leaf,
            seed: self.seed,
        }
    }
}
