#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub dim: usize,
}

impl StandardScaler {
    /// Fits mean and standard deviation on training matrix X (N x dim).
    pub fn fit(x: &[f64], num_samples: usize, dim: usize) -> Self {
        assert_eq!(x.len(), num_samples * dim, "Dimension mismatch");
        assert!(num_samples > 1, "Must have at least 2 samples to scale");

        let mut sum = vec![0.0; dim];
        for i in 0..num_samples {
            let row = &x[i * dim..(i + 1) * dim];
            for (d, item) in row.iter().enumerate().take(dim) {
                sum[d] += item;
            }
        }

        let mut mean = vec![0.0; dim];
        for (d, item) in mean.iter_mut().enumerate().take(dim) {
            *item = sum[d] / (num_samples as f64);
        }

        let mut sq_diff_sum = vec![0.0; dim];
        for i in 0..num_samples {
            let row = &x[i * dim..(i + 1) * dim];
            for (d, item) in row.iter().enumerate().take(dim) {
                let diff = item - mean[d];
                sq_diff_sum[d] += diff * diff;
            }
        }

        let mut std = vec![1.0; dim];
        for (d, item) in std.iter_mut().enumerate().take(dim) {
            let variance = sq_diff_sum[d] / (num_samples as f64);
            let s = variance.sqrt();
            *item = if s > 1e-8 { s } else { 1.0 };
        }

        Self { mean, std, dim }
    }

    /// Transforms matrix X into standardized coordinates (z-score normalization).
    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        let num_samples = x.len() / self.dim;
        let mut out = Vec::with_capacity(x.len());

        for i in 0..num_samples {
            let row = &x[i * self.dim..(i + 1) * self.dim];
            for (d, item) in row.iter().enumerate().take(self.dim) {
                out.push((item - self.mean[d]) / self.std[d]);
            }
        }

        out
    }

    /// Inverse transforms standardized coordinates back to original physical units.
    pub fn inverse_transform(&self, x_scaled: &[f64]) -> Vec<f64> {
        let num_samples = x_scaled.len() / self.dim;
        let mut out = Vec::with_capacity(x_scaled.len());

        for i in 0..num_samples {
            let row = &x_scaled[i * self.dim..(i + 1) * self.dim];
            for (d, item) in row.iter().enumerate().take(self.dim) {
                out.push(item * self.std[d] + self.mean[d]);
            }
        }

        out
    }
}

#[derive(Debug, Clone)]
pub struct TargetScaler {
    pub mean: f64,
    pub std: f64,
}

impl TargetScaler {
    pub fn fit(y: &[f64]) -> Self {
        let n = y.len();
        assert!(n > 1, "Must have at least 2 samples to scale targets");

        let mean = y.iter().sum::<f64>() / (n as f64);
        let sq_diff_sum = y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>();
        let variance = sq_diff_sum / (n as f64);
        let std_val = variance.sqrt();
        let std = if std_val > 1e-8 { std_val } else { 1.0 };

        Self { mean, std }
    }

    pub fn transform(&self, y: &[f64]) -> Vec<f64> {
        y.iter().map(|&v| (v - self.mean) / self.std).collect()
    }

    pub fn inverse_transform(&self, y_scaled: &[f64]) -> Vec<f64> {
        y_scaled.iter().map(|&v| v * self.std + self.mean).collect()
    }

    pub fn inverse_transform_scalar(&self, val_scaled: f64) -> f64 {
        val_scaled * self.std + self.mean
    }

    pub fn inverse_transform_variance(&self, var_scaled: f64) -> f64 {
        var_scaled * (self.std * self.std)
    }
}
