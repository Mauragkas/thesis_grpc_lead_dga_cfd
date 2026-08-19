use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub enum KernelType {
    Matern52 = 0,
    Rbf = 1,
}

/// Gaussian Process Hyperparameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpHyperparameters {
    pub length_scales: Vec<f64>,
    pub signal_variance: f64,
    pub noise_variance: f64,
    pub kernel_type: KernelType,
}

impl GpHyperparameters {
    pub fn new(
        length_scales: Vec<f64>,
        signal_variance: f64,
        noise_variance: f64,
        kernel_type: KernelType,
    ) -> Self {
        Self {
            length_scales,
            signal_variance,
            noise_variance,
            kernel_type,
        }
    }

    /// Creates default isotropic or unit length scales for dimension D.
    pub fn default_for_dim(dim: usize) -> Self {
        Self {
            length_scales: vec![1.0; dim],
            signal_variance: 1.0,
            noise_variance: 1e-3,
            kernel_type: KernelType::Matern52,
        }
    }
}
