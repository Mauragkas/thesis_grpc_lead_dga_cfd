//! Benchmark executing 5 independent random seeds of GaRunner (pop_size=20, max_generations=40)
//! Recording generation-by-generation best, average, and worst fitness to compute
//! Mean +/- Standard Deviation ribbons verifying statistical significance.

use async_trait::async_trait;
use orchestrator::config::GaConfig;
use orchestrator::evaluator::Evaluator;
use orchestrator::ga::algorithm::GaRunner;
use orchestrator::ga::telemetry::{GenerationRecord, TelemetrySink};
use orchestrator::gene_store::{GeneRecord, GeneStore};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

struct BenchmarkEvaluator;

#[async_trait]
impl Evaluator for BenchmarkEvaluator {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
        let mut results = Vec::with_capacity(population.len());
        for ind in population {
            let dist_sq: f64 = ind.iter().map(|&x| (x - 0.5).powi(2)).sum();
            let fitness = 15.0 / (1.0 + 3.0 * dist_sq);
            results.push(fitness);
        }
        Ok(results)
    }
}

struct SimpleStore;

#[async_trait]
impl GeneStore for SimpleStore {
    async fn store(&self, _genes: Vec<f64>, _fitness: f64, _generation: usize) {}
    async fn query_knn(&self, _query: &[f64], k: usize, _generation: usize) -> Vec<GeneRecord> {
        Vec::with_capacity(k)
    }
    async fn lookup_exact(&self, _genes: &[f64], _generation: usize) -> Option<f64> {
        None
    }
    async fn evict_expired(&self, _current_gen: usize) {}
}

pub struct InMemSink {
    pub records: Mutex<Vec<GenerationRecord>>,
}

impl InMemSink {
    pub fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl TelemetrySink for InMemSink {
    async fn record(&self, record: GenerationRecord) {
        let mut r = self.records.lock().await;
        r.push(record);
    }
}

#[derive(Serialize)]
struct SeedRunRecord {
    seed: u64,
    generations: Vec<usize>,
    best_fitness: Vec<f64>,
    avg_fitness: Vec<f64>,
}

#[tokio::main]
async fn main() {
    println!("Running 5 independent seeds for statistical significance ribbon...");
    let seeds = vec![42, 101, 2024, 777, 9999];
    let pop_size = 20;
    let max_gens = 40;
    let gene_len = 10;

    let evaluator = Arc::new(BenchmarkEvaluator);
    let store = Arc::new(SimpleStore);

    let mut all_runs = Vec::new();

    for &s in &seeds {
        let sink = Arc::new(InMemSink::new());
        let mut rng = StdRng::seed_from_u64(s);

        let ga_cfg = GaConfig {
            pop_size,
            genes_len: gene_len,
            max_generations: max_gens,
            min_generations: 5,
            stagnation_patience: 0,
            min_improvement: 0.001,
            mut_sigma: 0.10,
            elite_frac: 0.20,
            batch_size: 1,
            seed: s,
            eval_endpoint: "unused".to_string(),
            export_path: None,
            record_population: false,
        };

        let runner = GaRunner::new(&ga_cfg, evaluator.as_ref(), store.as_ref())
            .with_telemetry(Some(sink.as_ref()));

        runner.run(&mut rng).await.unwrap();

        let recs = sink.records.lock().await;
        let mut gens = Vec::new();
        let mut bests = Vec::new();
        let mut avgs = Vec::new();

        for r in recs.iter() {
            gens.push(r.generation);
            bests.push(r.best_fitness);
            avgs.push(r.avg_fitness);
        }

        all_runs.push(SeedRunRecord {
            seed: s,
            generations: gens,
            best_fitness: bests,
            avg_fitness: avgs,
        });
    }

    let json_text = serde_json::to_string_pretty(&all_runs).unwrap();
    let out_p = "../evaluation/data/real_multiseed_runs.json";
    let mut file = File::create(out_p).unwrap();
    file.write_all(json_text.as_bytes()).unwrap();
    println!("Saved multi-seed empirical runs to {out_p}");
}
