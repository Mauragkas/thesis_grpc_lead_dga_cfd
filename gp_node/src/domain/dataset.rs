use super::config::AircraftConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AeroMetrics {
    pub ld: Option<f64>,
    pub alpha_trim: Option<f64>,
    pub cm_alpha: Option<f64>,
    pub cl_req: Option<f64>,
    pub mass_g: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub config: AircraftConfig,
    pub aero: Option<AeroMetrics>,
    pub fitness: f64,
}

/// A contiguous 2D feature matrix (row-major) and target vector.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetPartition {
    pub x: Vec<f64>,       // Flattened row-major matrix of size [num_samples * dim]
    pub y: Vec<f64>,       // Target fitness vector of size [num_samples]
    pub num_samples: usize,
    pub dim: usize,
}

impl DatasetPartition {
    pub fn new(x: Vec<f64>, y: Vec<f64>, num_samples: usize, dim: usize) -> Self {
        assert_eq!(x.len(), num_samples * dim, "X length must equal num_samples * dim");
        assert_eq!(y.len(), num_samples, "y length must equal num_samples");
        Self { x, y, num_samples, dim }
    }

    pub fn get_row(&self, index: usize) -> &[f64] {
        let start = index * self.dim;
        &self.x[start..start + self.dim]
    }
}

/// 60-20-20 partition of training, validation, and test data.
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    pub train: DatasetPartition,
    pub validation: DatasetPartition,
    pub test: DatasetPartition,
}
