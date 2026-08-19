use crate::backend::traits::ComputeBackend;
use crate::domain::{GpHyperparameters, KernelType};

pub struct HyperparameterOptimizer;

impl HyperparameterOptimizer {
    /// Tunes length scales, signal variance, and noise variance to maximize Marginal Log-Likelihood.
    pub fn optimize_mll(
        backend: &dyn ComputeBackend,
        x_train: &[f64],
        y_train: &[f64],
        num_samples: usize,
        dim: usize,
        kernel_type: KernelType,
        max_iterations: usize,
    ) -> GpHyperparameters {
        // Multi-start candidates: isotropic length scales
        let initial_scales = [0.5, 1.0, 2.0, 3.5, 5.0];
        let mut best_params = GpHyperparameters {
            length_scales: vec![1.5; dim],
            signal_variance: 1.0,
            noise_variance: 1e-3,
            kernel_type,
        };

        let mut global_best_mll = f64::NEG_INFINITY;

        for &init_ls in &initial_scales {
            let params = GpHyperparameters {
                length_scales: vec![init_ls; dim],
                signal_variance: 1.0,
                noise_variance: 1e-3,
                kernel_type,
            };

            let mll = Self::evaluate_mll(backend, x_train, y_train, num_samples, dim, &params)
                .unwrap_or(f64::NEG_INFINITY);

            if mll > global_best_mll {
                global_best_mll = mll;
                best_params = params;
            }
        }

        let scale_factors = [0.5, 0.72, 0.88, 0.96, 1.04, 1.14, 1.38, 2.0];

        for _iter in 0..max_iterations {
            let mut improved = false;

            // 1. Coordinate descent over each dimension's length scale
            for d in 0..dim {
                let orig_ls = best_params.length_scales[d];
                let mut best_ls = orig_ls;
                let mut local_best_mll = global_best_mll;

                for &factor in &scale_factors {
                    let cand_ls = (orig_ls * factor).clamp(0.05, 50.0);
                    let mut cand_params = best_params.clone();
                    cand_params.length_scales[d] = cand_ls;

                    if let Some(mll) = Self::evaluate_mll(
                        backend,
                        x_train,
                        y_train,
                        num_samples,
                        dim,
                        &cand_params,
                    ) {
                        if mll > local_best_mll {
                            local_best_mll = mll;
                            best_ls = cand_ls;
                        }
                    }
                }

                if local_best_mll > global_best_mll + 1e-4 {
                    best_params.length_scales[d] = best_ls;
                    global_best_mll = local_best_mll;
                    improved = true;
                }
            }

            // 2. Tune signal variance
            let orig_sig = best_params.signal_variance;
            for &factor in &scale_factors {
                let cand_sig = (orig_sig * factor).clamp(0.05, 30.0);
                let mut cand_params = best_params.clone();
                cand_params.signal_variance = cand_sig;

                if let Some(mll) =
                    Self::evaluate_mll(backend, x_train, y_train, num_samples, dim, &cand_params)
                {
                    if mll > global_best_mll + 1e-4 {
                        best_params.signal_variance = cand_sig;
                        global_best_mll = mll;
                        improved = true;
                    }
                }
            }

            // 3. Tune noise variance
            let orig_noise = best_params.noise_variance;
            for &factor in &[0.3, 0.6, 0.85, 1.15, 1.6, 3.0] {
                let cand_noise = (orig_noise * factor).clamp(1e-6, 0.5);
                let mut cand_params = best_params.clone();
                cand_params.noise_variance = cand_noise;

                if let Some(mll) =
                    Self::evaluate_mll(backend, x_train, y_train, num_samples, dim, &cand_params)
                {
                    if mll > global_best_mll + 1e-4 {
                        best_params.noise_variance = cand_noise;
                        global_best_mll = mll;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }

        best_params
    }

    fn evaluate_mll(
        backend: &dyn ComputeBackend,
        x: &[f64],
        y: &[f64],
        n: usize,
        dim: usize,
        params: &GpHyperparameters,
    ) -> Option<f64> {
        let k = backend
            .compute_covariance(x, n, x, n, dim, params, true)
            .ok()?;
        let l = backend.cholesky(&k, n).ok()?;
        let alpha = backend.compute_alpha(&l, y, n).ok()?;
        backend.log_marginal_likelihood(y, &alpha, &l, n).ok()
    }
}
