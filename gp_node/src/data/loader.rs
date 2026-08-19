use crate::domain::DatasetRecord;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("IO error while reading dataset file: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON deserialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("No valid dataset records found after filtering")]
    EmptyDataset,
}

pub struct DatasetLoader;

impl DatasetLoader {
    /// Loads dataset records from a JSON file, sanitizes Python NaN/Infinity literals to null,
    /// and filters out infeasible / unconverged designs.
    pub fn load_from_json<P: AsRef<Path>>(
        filepath: P,
        min_fitness: Option<f64>,
    ) -> Result<Vec<DatasetRecord>, LoaderError> {
        let content = fs::read_to_string(filepath)?;
        // Sanitize non-standard JSON literals produced by Python json.dump
        let sanitized = content
            .replace(": NaN", ": null")
            .replace(": Infinity", ": null")
            .replace(": -Infinity", ": null");

        let records: Vec<DatasetRecord> = serde_json::from_str(&sanitized)?;

        let min_fit = min_fitness.unwrap_or(-15.0);
        let valid_records: Vec<DatasetRecord> = records
            .into_iter()
            .filter(|r| r.fitness > min_fit && !r.fitness.is_nan())
            .collect();

        if valid_records.is_empty() {
            return Err(LoaderError::EmptyDataset);
        }

        Ok(valid_records)
    }

    /// Converts records to a raw matrix X and target vector y using active 11 features.
    pub fn extract_active_features(records: &[DatasetRecord]) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let n = records.len();
        let dim = 11;
        let mut x = Vec::with_capacity(n * dim);
        let mut y = Vec::with_capacity(n);

        for r in records {
            x.extend(r.config.to_active_features());
            y.push(r.fitness);
        }

        (x, y, n, dim)
    }

    /// Converts records to a raw matrix X and target vector y using all 22 config parameters.
    pub fn extract_all_features(records: &[DatasetRecord]) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let n = records.len();
        let dim = 22;
        let mut x = Vec::with_capacity(n * dim);
        let mut y = Vec::with_capacity(n);

        for r in records {
            x.extend(r.config.to_all_features());
            y.push(r.fitness);
        }

        (x, y, n, dim)
    }
}
