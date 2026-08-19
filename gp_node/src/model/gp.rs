use crate::backend::traits::ComputeBackend;
use crate::backend::ffi::BackendError;
use crate::data::scaler::{StandardScaler, TargetScaler};
use crate::domain::{EvaluationMetrics, GpHyperparameters, KernelType};
use super::optimizer::HyperparameterOptimizer;

pub struct GaussianProcessSurrogate {
    backend: Box<dyn ComputeBackend>,
    params: GpHyperparameters,
    x_scaler: Option<StandardScaler>,
    y_scaler: Option<TargetScaler>,
    train_x_scaled: Vec<f64>,
    train_y_scaled: Vec<f64>,
    alpha: Vec<f64>,
    l_factor: Vec<f64>,
    n_train: usize,
    dim: usize,
}

impl GaussianProcessSurrogate {
    pub fn new(backend: Box<dyn ComputeBackend>, kernel_type: KernelType) -> Self {
        Self {
            backend,
            params: GpHyperparameters::new(vec![], 1.0, 1e-3, kernel_type),
            x_scaler: None,
            y_scaler: None,
            train_x_scaled: Vec::new(),
            train_y_scaled: Vec::new(),
            alpha: Vec::new(),
            l_factor: Vec::new(),
            n_train: 0,
            dim: 0,
        }
    }

    pub fn backend(&self) -> &dyn ComputeBackend {
        self.backend.as_ref()
    }

    pub fn hyperparameters(&self) -> &GpHyperparameters {
        &self.params
    }

    pub fn set_hyperparameters(&mut self, params: GpHyperparameters) {
        self.params = params;
    }

    /// Fits the Gaussian Process model on training data X (N x dim) and targets y (N).
    pub fn fit(
        &mut self,
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        dim: usize,
        optimize_hyperparams: bool,
    ) -> Result<(), BackendError> {
        assert_eq!(x_train.len(), n_train * dim, "x_train length mismatch");
        assert_eq!(y_train.len(), n_train, "y_train length mismatch");

        self.n_train = n_train;
        self.dim = dim;

        // 1. Fit standard scalers on training partition
        let x_scaler = StandardScaler::fit(x_train, n_train, dim);
        let y_scaler = TargetScaler::fit(y_train);

        let x_scaled = x_scaler.transform(x_train);
        let y_scaled = y_scaler.transform(y_train);

        // 2. Hyperparameter optimization if requested
        if optimize_hyperparams || self.params.length_scales.len() != dim {
            self.params = HyperparameterOptimizer::optimize_mll(
                self.backend.as_ref(),
                &x_scaled,
                &y_scaled,
                n_train,
                dim,
                self.params.kernel_type,
                15,
            );
        }

        // 3. Compute covariance matrix K + sigma_n^2 * I
        let k = self.backend.compute_covariance(
            &x_scaled,
            n_train,
            &x_scaled,
            n_train,
            dim,
            &self.params,
            true,
        )?;

        // 4. Cholesky factorization L * L^T = K
        let l = self.backend.cholesky(&k, n_train)?;

        // 5. Compute alpha weights
        let alpha = self.backend.compute_alpha(&l, &y_scaled, n_train)?;

        self.x_scaler = Some(x_scaler);
        self.y_scaler = Some(y_scaler);
        self.train_x_scaled = x_scaled;
        self.train_y_scaled = y_scaled;
        self.alpha = alpha;
        self.l_factor = l;

        Ok(())
    }

    /// Predicts target mean and standard deviation for raw test matrix X_test.
    pub fn predict(&self, x_test: &[f64]) -> Result<(Vec<f64>, Vec<f64>), BackendError> {
        let x_scaler = self.x_scaler.as_ref().expect("Model must be fitted before predict");
        let y_scaler = self.y_scaler.as_ref().expect("Model must be fitted before predict");

        let n_test = x_test.len() / self.dim;
        assert_eq!(x_test.len(), n_test * self.dim);

        let x_test_scaled = x_scaler.transform(x_test);

        let (mean_scaled, var_scaled) = self.backend.predict_batch(
            &self.train_x_scaled,
            self.n_train,
            &x_test_scaled,
            n_test,
            self.dim,
            &self.alpha,
            &self.l_factor,
            &self.params,
        )?;

        let mut mean_orig = Vec::with_capacity(n_test);
        let mut std_orig = Vec::with_capacity(n_test);

        for (m_s, v_s) in mean_scaled.into_iter().zip(var_scaled) {
            mean_orig.push(y_scaler.inverse_transform_scalar(m_s));
            let v_orig = y_scaler.inverse_transform_variance(v_s);
            std_orig.push(v_orig.max(0.0).sqrt());
        }

        Ok((mean_orig, std_orig))
    }

    /// Evaluates model on a test set and returns regression metrics (R2, RMSE, MAE).
    pub fn evaluate(&self, x_test: &[f64], y_test: &[f64]) -> Result<EvaluationMetrics, BackendError> {
        let (y_pred, _) = self.predict(x_test)?;
        Ok(EvaluationMetrics::compute(y_test, &y_pred))
    }
}
