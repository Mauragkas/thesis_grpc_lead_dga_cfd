use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use tracing::{error, info};

/// Structured generational record capturing all statistics required for thesis plots.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerationRecord {
    pub generation: usize,
    pub elapsed_sec: f64,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub worst_fitness: f64,
    pub std_fitness: f64,
    pub best_genome: Vec<f64>,
    /// Genetic variance computed across the population for each gene dimension.
    pub gene_variance: Vec<f64>,
    /// Normalized population entropy (Shannon diversity index) in range [0.0, 1.0].
    pub entropy: f64,
    /// Optional full population snapshot (only when full population logging is enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub population: Option<Vec<Vec<f64>>>,
    /// Generational Tier 1 exact cache hits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier1_exact_hits: Option<u64>,
    /// Generational Tier 2 surrogate model evaluation hits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier2_surrogate_hits: Option<u64>,
    /// Generational Tier 3 expensive simulator/CFD evaluations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier3_cfd_evals: Option<u64>,
    /// Generational simulation bypass ratio: (Tier 1 + Tier 2) / Total evaluations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_bypass_ratio: Option<f64>,
}

/// Computes per-gene variance across the population.
/// Returns a vector of length `genes_len` with sample variance (or population variance if N < 2).
pub fn compute_gene_variance(population: &[Vec<f64>], genes_len: usize) -> Vec<f64> {
    if population.is_empty() || genes_len == 0 {
        return vec![0.0; genes_len];
    }
    let n = population.len() as f64;
    let mut means = vec![0.0; genes_len];
    for ind in population {
        for (j, &val) in ind.iter().enumerate().take(genes_len) {
            means[j] += val;
        }
    }
    for m in &mut means {
        *m /= n;
    }

    let mut variances = vec![0.0; genes_len];
    for ind in population {
        for (j, &val) in ind.iter().enumerate().take(genes_len) {
            let diff = val - means[j];
            variances[j] += diff * diff;
        }
    }

    let divisor = if population.len() > 1 {
        (population.len() - 1) as f64
    } else {
        1.0
    };
    for v in &mut variances {
        *v /= divisor;
    }
    variances
}

/// Computes normalized Shannon entropy of population diversity based on discretized binning.
/// Maps normalized gene values [0.0, 1.0] into `num_bins` per dimension.
pub fn compute_population_entropy(
    population: &[Vec<f64>],
    genes_len: usize,
    num_bins: usize,
) -> f64 {
    if population.is_empty() || genes_len == 0 || num_bins <= 1 {
        return 0.0;
    }
    let n = population.len() as f64;
    let mut total_dim_entropy = 0.0;

    for dim in 0..genes_len {
        let mut bin_counts = vec![0usize; num_bins];
        for ind in population {
            if let Some(&val) = ind.get(dim) {
                let clamped = val.clamp(0.0, 1.0);
                let mut bin = (clamped * num_bins as f64).floor() as usize;
                if bin >= num_bins {
                    bin = num_bins - 1;
                }
                bin_counts[bin] += 1;
            }
        }

        let mut dim_entropy = 0.0;
        for &count in &bin_counts {
            if count > 0 {
                let p = count as f64 / n;
                dim_entropy -= p * p.ln();
            }
        }
        // Normalize by max possible entropy ln(num_bins)
        let max_entropy = (num_bins as f64).ln();
        if max_entropy > 0.0 {
            dim_entropy /= max_entropy;
        }
        total_dim_entropy += dim_entropy;
    }

    total_dim_entropy / genes_len as f64
}

/// Computes fitness statistics: (best, avg, worst, std_dev)
pub fn compute_fitness_stats(fitnesses: &[f64]) -> (f64, f64, f64, f64) {
    if fitnesses.is_empty() {
        return (f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY, 0.0);
    }
    let mut best = f64::NEG_INFINITY;
    let mut worst = f64::INFINITY;
    let mut sum = 0.0;

    for &f in fitnesses {
        if f > best {
            best = f;
        }
        if f < worst {
            worst = f;
        }
        sum += f;
    }

    let avg = sum / fitnesses.len() as f64;
    let var_sum: f64 = fitnesses.iter().map(|&f| (f - avg).powi(2)).sum();
    let std_dev = (var_sum / fitnesses.len() as f64).sqrt();

    (best, avg, worst, std_dev)
}

/// Abstract telemetry sink (DIP - Dependency Inversion Principle).
#[async_trait]
pub trait TelemetrySink: Send + Sync {
    async fn record(&self, record: GenerationRecord);
}

/// No-op sink when telemetry is disabled.
pub struct NoopTelemetrySink;

#[async_trait]
impl TelemetrySink for NoopTelemetrySink {
    async fn record(&self, _record: GenerationRecord) {}
}

/// Thread-safe JSON Lines file writer sink.
pub struct JsonLinesFileSink {
    file: Mutex<File>,
}

impl JsonLinesFileSink {
    pub fn try_new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(p)?;
        info!("GA telemetry exporter initialized at {:?}", p);
        Ok(Self {
            file: Mutex::new(file),
        })
    }
}

#[async_trait]
impl TelemetrySink for JsonLinesFileSink {
    async fn record(&self, record: GenerationRecord) {
        match serde_json::to_string(&record) {
            Ok(json) => {
                if let Ok(mut lock) = self.file.lock() {
                    if let Err(e) = writeln!(lock, "{}", json) {
                        error!("Failed to write telemetry record: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to serialize telemetry record: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fitness_stats() {
        let fitnesses = vec![10.0, 20.0, 30.0];
        let (best, avg, worst, std_dev) = compute_fitness_stats(&fitnesses);
        assert_eq!(best, 30.0);
        assert_eq!(worst, 10.0);
        assert_eq!(avg, 20.0);
        assert!((std_dev - 8.1649).abs() < 0.001);
    }

    #[test]
    fn test_compute_gene_variance() {
        let pop = vec![
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let variances = compute_gene_variance(&pop, 2);
        assert_eq!(variances.len(), 2);
        assert!((variances[0] - 0.5).abs() < 1e-6);
        assert!((variances[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_population_entropy() {
        // Uniformly collapsed population should have 0 entropy
        let pop_uniform = vec![vec![0.5, 0.5]; 10];
        let ent_uniform = compute_population_entropy(&pop_uniform, 2, 10);
        assert_eq!(ent_uniform, 0.0);

        // Diverse population across 10 bins should have higher entropy
        let mut pop_diverse = Vec::new();
        for i in 0..10 {
            let val = (i as f64 + 0.5) / 10.0;
            pop_diverse.push(vec![val, val]);
        }
        let ent_diverse = compute_population_entropy(&pop_diverse, 2, 10);
        assert!((ent_diverse - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_json_lines_sink() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_ga_telemetry.jsonl");
        let sink = JsonLinesFileSink::try_new(&path).unwrap();

        let record = GenerationRecord {
            generation: 1,
            elapsed_sec: 0.12,
            best_fitness: 42.5,
            avg_fitness: 30.0,
            worst_fitness: 10.0,
            std_fitness: 5.0,
            best_genome: vec![0.1, 0.2],
            gene_variance: vec![0.01, 0.02],
            entropy: 0.75,
            population: None,
            tier1_exact_hits: Some(15),
            tier2_surrogate_hits: Some(45),
            tier3_cfd_evals: Some(40),
            tier_bypass_ratio: Some(0.60),
        };

        sink.record(record.clone()).await;

        let content = std::fs::read_to_string(&path).unwrap();
        let deserialized: GenerationRecord = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(deserialized, record);

        let _ = std::fs::remove_file(path);
    }
}
