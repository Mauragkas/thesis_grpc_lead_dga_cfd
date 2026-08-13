#![allow(unused)]
//! Shared test doubles. Each mock implements a crate trait (DIP) so the
//! same fakes can be reused across test files without duplicating logic.

use async_trait::async_trait;
use orchestrator::evaluator::Evaluator;
use orchestrator::gene_store::{GeneRecord, GeneStore};
use orchestrator::migration::{MigrantIndividual, MigrationHook};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tonic::Status;

/// Deterministic evaluator: returns a preconfigured fitness vector,
/// optionally fails, and counts calls. `Send + Sync` via atomics/Mutex.
pub struct MockEvaluator {
    pub fitnesses: Mutex<Vec<f64>>,
    pub calls: AtomicUsize,
    pub fail: bool,
}

impl MockEvaluator {
    pub fn new(fitnesses: Vec<f64>) -> Self {
        Self {
            fitnesses: Mutex::new(fitnesses),
            calls: AtomicUsize::new(0),
            fail: false,
        }
    }

    pub fn failing() -> Self {
        Self {
            fitnesses: Mutex::new(vec![]),
            calls: AtomicUsize::new(0),
            fail: true,
        }
    }

    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Evaluator for MockEvaluator {
    async fn evaluate_population(&self, pop: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(Status::internal("mock evaluator failure"));
        }
        let f = self.fitnesses.lock().unwrap();
        Ok(pop
            .iter()
            .enumerate()
            .map(|(i, _)| f.get(i).copied().unwrap_or(0.0))
            .collect())
    }
}

/// Configurable in-memory store that can force cache hits and record
/// eviction calls. Independent from the real `InMemoryGeneStore` so
/// algorithm tests stay isolated from storage internals.
pub struct MockGeneStore {
    pub exact: Mutex<HashMap<String, f64>>,
    pub store_calls: AtomicUsize,
    pub evict_calls: AtomicUsize,
    pub knn_calls: AtomicUsize,
}

impl MockGeneStore {
    pub fn empty() -> Self {
        Self {
            exact: Mutex::new(HashMap::new()),
            store_calls: AtomicUsize::new(0),
            evict_calls: AtomicUsize::new(0),
            knn_calls: AtomicUsize::new(0),
        }
    }

    /// Pre-seed an exact-match hit for the given gene vector.
    pub fn with_exact(genes: &[f64], fitness: f64) -> Self {
        let s = Self::empty();
        s.seed_exact(genes, fitness);
        s
    }

    pub fn seed_exact(&self, genes: &[f64], fitness: f64) {
        let key = genes
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");
        self.exact.lock().unwrap().insert(key, fitness);
    }
}

#[async_trait]
impl GeneStore for MockGeneStore {
    async fn store(&self, genes: Vec<f64>, fitness: f64, generation: usize) {
        self.store_calls.fetch_add(1, Ordering::SeqCst);
        let key = genes
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");
        self.exact.lock().unwrap().insert(key, fitness);
        let _ = generation;
    }

    async fn query_knn(
        &self,
        _query: &[f64],
        k: usize,
        _current_generation: usize,
    ) -> Vec<GeneRecord> {
        self.knn_calls.fetch_add(1, Ordering::SeqCst);
        Vec::with_capacity(k)
    }

    async fn lookup_exact(&self, genes: &[f64], _current_generation: usize) -> Option<f64> {
        let key = genes
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");
        self.exact.lock().unwrap().get(&key).copied()
    }

    async fn evict_expired(&self, _current_generation: usize) {
        self.evict_calls.fetch_add(1, Ordering::SeqCst);
    }
}

/// No-op migration hook for tests.
pub struct NoopMigration;

#[async_trait]
impl MigrationHook for NoopMigration {
    async fn maybe_emigrate(
        &self,
        _generation: usize,
        _population: &[Vec<f64>],
        _fitnesses: &[f64],
    ) -> Result<(), tonic::Status> {
        Ok(())
    }

    async fn drain_immigrants(&self) -> Vec<MigrantIndividual> {
        Vec::new()
    }
}

/// Build a small `GaConfig` for deterministic, fast tests.
pub fn small_config() -> orchestrator::config::GaConfig {
    orchestrator::config::GaConfig {
        pop_size: 6,
        genes_len: 4,
        generations: 3,
        mut_sigma: 0.05,
        elite_frac: 0.5,
        batch_size: 2,
        seed: 7,
        eval_endpoint: "127.0.0.1:0".to_string(),
    }
}
