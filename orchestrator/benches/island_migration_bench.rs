//! Integration benchmark running real multi-island Genetic Algorithm instances
//! with small populations (pop_size=20, 4 islands) across different MigrationConfig settings:
//!   - M_interval in [2, 5, 15] generations
//!   - M_count in [1, 3] migrants
//! Measures real generation-by-generation fitness progression and inter-island entropy!

use async_trait::async_trait;
use orchestrator::config::GaConfig;
use orchestrator::evaluator::Evaluator;
use orchestrator::ga::algorithm::GaRunner;
use orchestrator::ga::telemetry::{GenerationRecord, TelemetrySink};
use orchestrator::gene_store::{GeneRecord, GeneStore};
use orchestrator::migration::buffer::MigrantBuffer;
use orchestrator::migration::config::MigrationConfig;
use orchestrator::migration::{LeadMigration, MigrantIndividual, TopKSelector};
use orchestrator::ring::client::RingClient;
use orchestrator::ring::member::RingMember;
use orchestrator::ring::state::NodeInfo;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

/// Mathematical fitness evaluator (Sphere landscape in [0, 1]^D, optimum at 0.5)
struct FastBenchmarkEvaluator;

#[async_trait]
impl Evaluator for FastBenchmarkEvaluator {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
        let mut results = Vec::with_capacity(population.len());
        for ind in population {
            let dist_sq: f64 = ind.iter().map(|&x| (x - 0.5).powi(2)).sum();
            // Optimum is 15.0 when all genes = 0.5
            let fitness = 15.0 / (1.0 + 3.0 * dist_sq);
            results.push(fitness);
        }
        Ok(results)
    }
}

/// In-memory GeneStore
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

/// In-memory Telemetry Sink to capture each generation record
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

/// Shared Ring Network interconnecting 4 in-memory islands
struct SharedRingNetwork {
    buffers: Mutex<HashMap<String, Arc<MigrantBuffer>>>,
}

struct RingNodeMember {
    self_info: NodeInfo,
    successor: NodeInfo,
}

#[async_trait]
impl RingMember for RingNodeMember {
    async fn find_successor(&self, _id: u64) -> NodeInfo {
        self.successor.clone()
    }
    async fn get_predecessor(&self) -> Option<NodeInfo> {
        None
    }
    async fn get_successor(&self) -> Option<NodeInfo> {
        Some(self.successor.clone())
    }
    fn self_node(&self) -> NodeInfo {
        self.self_info.clone()
    }
    async fn notify(&self, _other: NodeInfo) -> bool {
        true
    }
    async fn get_successor_list(&self) -> Vec<NodeInfo> {
        vec![self.successor.clone()]
    }
    async fn set_successor(&self, _node: NodeInfo) {}
    async fn set_successor_list(&self, _list: Vec<NodeInfo>) {}
}

struct DirectRingClient {
    network: Arc<SharedRingNetwork>,
}

#[async_trait]
impl RingClient for DirectRingClient {
    async fn find_successor(&self, _addr: &str, _id: u64) -> Result<NodeInfo, Status> {
        Ok(NodeInfo::new(1, "node"))
    }
    async fn get_predecessor(&self, _addr: &str) -> Result<Option<NodeInfo>, Status> {
        Ok(None)
    }
    async fn notify(&self, _addr: &str, _other: NodeInfo) -> Result<bool, Status> {
        Ok(true)
    }
    async fn get_successor_list(&self, _addr: &str) -> Result<Vec<NodeInfo>, Status> {
        Ok(vec![])
    }
    async fn ping(&self, _addr: &str) -> Result<bool, Status> {
        Ok(true)
    }
    async fn migrate(
        &self,
        addr: &str,
        _sender: &str,
        migrants: &[MigrantIndividual],
    ) -> Result<bool, Status> {
        let buffers = self.network.buffers.lock().await;
        if let Some(buf) = buffers.get(addr) {
            buf.push(migrants.to_vec()).await;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Controlled stepping evaluator that yields execution to allow inter-island concurrency
struct SteppingEvaluator {
    inner: FastBenchmarkEvaluator,
}

#[async_trait]
impl Evaluator for SteppingEvaluator {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        self.inner.evaluate_population(population).await
    }
}

#[derive(Serialize)]
struct MigrationRunResult {
    interval: usize,
    count: usize,
    history_best: Vec<f64>,
    history_entropy: Vec<f64>,
    generations_to_converge: usize,
}

#[tokio::main]
async fn main() {
    println!("Starting real in-memory multi-island GA migration benchmarks...");
    let num_islands = 4;
    let pop_size = 20; // Small population
    let max_gens = 35;
    let gene_len = 10;

    let test_configs = vec![
        (2, 3),  // Frequent migration (M_int = 2)
        (5, 3),  // Moderate / thesis default (M_int = 5)
        (15, 3), // Rare migration (M_int = 15)
        (5, 1),  // Small migrant volume (count = 1)
    ];

    let evaluator = Arc::new(SteppingEvaluator {
        inner: FastBenchmarkEvaluator,
    });
    let store = Arc::new(SimpleStore);

    let mut benchmark_results = Vec::new();

    for (config_idx, (interval, count)) in test_configs.into_iter().enumerate() {
        println!("Running 4-island GA: interval={interval}, count={count}...");

        let network = Arc::new(SharedRingNetwork {
            buffers: Mutex::new(HashMap::new()),
        });

        let mut island_buffers = Vec::new();
        for i in 0..num_islands {
            let buf = Arc::new(MigrantBuffer::new());
            network
                .buffers
                .lock()
                .await
                .insert(format!("island_{i}"), buf.clone());
            island_buffers.push(buf);
        }

        let sinks: Vec<Arc<InMemSink>> = (0..num_islands)
            .map(|_| Arc::new(InMemSink::new()))
            .collect();

        let mut handles = Vec::new();

        for i in 0..num_islands {
            let net_ref = network.clone();
            let sink_ref = sinks[i].clone();
            let eval_ref = evaluator.clone();
            let store_ref = store.clone();
            let self_addr = format!("island_{i}");
            let succ_addr = format!("island_{}", (i + 1) % num_islands);

            let member = Arc::new(RingNodeMember {
                self_info: NodeInfo::new(i as u64, &self_addr),
                successor: NodeInfo::new(((i + 1) % num_islands) as u64, &succ_addr),
            });
            let client = Arc::new(DirectRingClient { network: net_ref });
            let buffer = island_buffers[i].clone();

            let mig_cfg = MigrationConfig {
                interval_generations: interval,
                migrant_count: count,
            };
            let migration_hook = Arc::new(LeadMigration::new(
                mig_cfg,
                member,
                client,
                TopKSelector,
                buffer.clone(),
            ));

            // Island-specific random seed
            let island_seed = (config_idx as u64 * 1000) + (i as u64 * 73) + 42;

            let ga_cfg = GaConfig {
                pop_size,
                genes_len: gene_len,
                max_generations: max_gens,
                min_generations: 5,
                stagnation_patience: 0,
                min_improvement: 0.001,
                mut_sigma: 0.08,
                elite_frac: 0.20,
                batch_size: 1,
                seed: island_seed,
                eval_endpoint: "unused".to_string(),
                export_path: None,
                record_population: false,
            };

            let h = tokio::spawn(async move {
                let mut rng = StdRng::seed_from_u64(ga_cfg.seed);
                let runner = GaRunner::new(&ga_cfg, eval_ref.as_ref(), store_ref.as_ref())
                    .with_migration(Some(migration_hook.as_ref()))
                    .with_telemetry(Some(sink_ref.as_ref()));
                runner.run(&mut rng).await.unwrap();
            });
            handles.push(h);
        }

        for h in handles {
            h.await.unwrap();
        }

        // Aggregate results
        let mut best_per_gen = vec![0.0f64; max_gens + 1];
        let mut ent_per_gen = vec![0.0f64; max_gens + 1];

        for gen in 1..=max_gens {
            let mut gen_bests = Vec::new();
            let mut gen_ents = Vec::new();
            for sink in &sinks {
                let records = sink.records.lock().await;
                if let Some(r) = records.iter().find(|rec| rec.generation == gen) {
                    gen_bests.push(r.best_fitness);
                    gen_ents.push(r.entropy);
                }
            }
            if !gen_bests.is_empty() {
                best_per_gen[gen] = gen_bests.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                ent_per_gen[gen] = gen_ents.iter().sum::<f64>() / gen_ents.len() as f64;
            }
        }

        let target_fit = 11.5;
        let conv_gen = best_per_gen
            .iter()
            .position(|&f| f >= target_fit)
            .unwrap_or(max_gens);

        println!(
            "Configuration (int={interval}, count={count}): conv_gen={conv_gen}, final_best={:.2}, final_entropy={:.3}",
            best_per_gen[max_gens], ent_per_gen[max_gens]
        );

        benchmark_results.push(MigrationRunResult {
            interval,
            count,
            history_best: best_per_gen,
            history_entropy: ent_per_gen,
            generations_to_converge: conv_gen,
        });
    }

    let json_text = serde_json::to_string_pretty(&benchmark_results).unwrap();
    let out_p = "../evaluation/data/real_migration_benchmarks.json";
    let mut file = File::create(out_p).unwrap();
    file.write_all(json_text.as_bytes()).unwrap();
    println!("Saved real migration benchmark results to {out_p}");
}
