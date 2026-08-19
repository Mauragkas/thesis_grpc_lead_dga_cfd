use crate::domain::{GpHyperparameters, KernelType};

/// Pure Rust reference implementations of covariance kernels for testing and verification.
pub struct ReferenceKernel;

impl ReferenceKernel {
    pub fn compute_dist_sq(x1: &[f64], x2: &[f64], length_scales: &[f64]) -> f64 {
        assert_eq!(x1.len(), x2.len());
        assert_eq!(x1.len(), length_scales.len());

        let mut sum = 0.0;
        for d in 0..x1.len() {
            let diff = (x1[d] - x2[d]) / length_scales[d];
            sum += diff * diff;
        }
        sum
    }

    pub fn eval_kernel(
        x1: &[f64],
        x2: &[f64],
        params: &GpHyperparameters,
    ) -> f64 {
        let d2 = Self::compute_dist_sq(x1, x2, &params.length_scales);
        match params.kernel_type {
            KernelType::Matern52 => {
                let dist = d2.max(0.0).sqrt();
                let sqrt5_r = 5.0f64.sqrt() * dist;
                params.signal_variance * (1.0 + sqrt5_r + (5.0 / 3.0) * d2) * (-sqrt5_r).exp()
            }
            KernelType::Rbf => {
                params.signal_variance * (-0.5 * d2).exp()
            }
        }
    }
}
