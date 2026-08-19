use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub r2_score: f64,
    pub rmse: f64,
    pub mae: f64,
    pub max_residual: f64,
    pub num_samples: usize,
}

impl EvaluationMetrics {
    /// Computes statistical regression metrics comparing ground truth y with predictions y_pred.
    pub fn compute(y_true: &[f64], y_pred: &[f64]) -> Self {
        assert_eq!(
            y_true.len(),
            y_pred.len(),
            "y_true and y_pred must have identical length"
        );
        let n = y_true.len();
        if n == 0 {
            return Self {
                r2_score: 0.0,
                rmse: 0.0,
                mae: 0.0,
                max_residual: 0.0,
                num_samples: 0,
            };
        }

        let y_mean = y_true.iter().sum::<f64>() / (n as f64);
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        let mut sum_abs_err = 0.0;
        let mut max_res = 0.0f64;

        for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
            let diff_mean = yt - y_mean;
            ss_tot += diff_mean * diff_mean;

            let residual = yt - yp;
            let abs_res = residual.abs();
            ss_res += residual * residual;
            sum_abs_err += abs_res;
            if abs_res > max_res {
                max_res = abs_res;
            }
        }

        let r2 = if ss_tot > 1e-12 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        let rmse = (ss_res / (n as f64)).sqrt();
        let mae = sum_abs_err / (n as f64);

        Self {
            r2_score: r2,
            rmse,
            mae,
            max_residual: max_res,
            num_samples: n,
        }
    }
}
