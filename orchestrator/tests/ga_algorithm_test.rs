//! Integration tests for `GaRunner`. Uses mock `Evaluator` and `GeneStore`
//! from `common` so the algorithm is tested in isolation from gRPC/storage.

mod common;

use common::{small_config, MockEvaluator, MockGeneStore};
use orchestrator::config::TierConfig;
use orchestrator::evaluator::MultiTierEvaluator;
use orchestrator::ga::algorithm::GaRunner;
use orchestrator::surrogate_client::MockSurrogateClient;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Arc;
use tonic::Status;

fn seeded_rng(cfg: &orchestrator::config::GaConfig) -> StdRng {
    StdRng::seed_from_u64(cfg.seed)
}

#[tokio::test]
async fn runner_returns_best_fitness_on_happy_path() {
    let cfg = small_config();
    let pop = cfg.pop_size;
    // descending fitness so best is at index 0
    let fitnesses: Vec<f64> = (0..pop).map(|i| (pop - i) as f64).collect();
    let evaluator = MockEvaluator::new(fitnesses);
    let store = MockGeneStore::empty();

    let mut rng = seeded_rng(&cfg);
    let runner = GaRunner {
        cfg: &cfg,
        evaluator: &evaluator,
        store: &store,
        neighbor_store: None,
        migration: None,
    };
    let best = runner.run(&mut rng).await.unwrap();
    // best ever should be the max of the supplied fitnesses
    assert!((best.best_fitness - (pop as f64)).abs() < 1e-9);
    assert_eq!(best.best_genome.len(), cfg.genes_len);
    assert!(evaluator.call_count() > 0);
}

#[tokio::test]
async fn runner_propagates_evaluator_errors() {
    let cfg = small_config();
    let evaluator = MockEvaluator::failing();
    let store = MockGeneStore::empty();

    let mut rng = seeded_rng(&cfg);
    let runner = GaRunner {
        cfg: &cfg,
        evaluator: &evaluator,
        store: &store,
        neighbor_store: None,
        migration: None,
    };
    let err = runner.run(&mut rng).await.unwrap_err();
    assert_eq!(err.code(), Status::internal("x").code());
    assert!(err.message().contains("mock evaluator failure"));
}

#[tokio::test]
async fn runner_with_multi_tier_evaluator_exact_cache_hits() {
    let cfg = small_config();
    let mut rng = seeded_rng(&cfg);
    let pop = orchestrator::ga::operators::random_population(&mut rng, &cfg);

    let store = Arc::new(MockGeneStore::empty());
    let half = pop.len() / 2;
    for g in pop.iter().take(half) {
        store.seed_exact(g, 100.0);
    }

    let sim = Arc::new(MockEvaluator::new((0..pop.len()).map(|i| i as f64).collect()));
    let surrogate = Arc::new(MockSurrogateClient::new(true, 50.0));
    let tier_cfg = TierConfig::default();

    let multi_tier = MultiTierEvaluator::new(
        sim,
        store.clone(),
        None,
        Some(surrogate),
        tier_cfg,
    );

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let runner = GaRunner {
        cfg: &cfg,
        evaluator: &multi_tier,
        store: store.as_ref(),
        neighbor_store: None,
        migration: None,
    };
    let best = runner.run(&mut rng).await.unwrap();
    assert!(best.best_fitness >= 100.0);
    assert_eq!(best.best_genome.len(), cfg.genes_len);
    let snap = multi_tier.metrics.snapshot();
    assert!(snap.tier1_exact_hits > 0);
}

#[tokio::test]
async fn runner_invokes_eviction_each_generation() {
    let cfg = small_config();
    let evaluator = MockEvaluator::new((0..cfg.pop_size).map(|i| i as f64).collect());
    let store = MockGeneStore::empty();

    let mut rng = seeded_rng(&cfg);
    let runner = GaRunner {
        cfg: &cfg,
        evaluator: &evaluator,
        store: &store,
        neighbor_store: None,
        migration: None,
    };
    runner.run(&mut rng).await.unwrap();
    assert_eq!(
        store.evict_calls.load(std::sync::atomic::Ordering::SeqCst),
        cfg.generations
    );
}

#[tokio::test]
async fn runner_with_multi_tier_stores_newly_evaluated_individuals() {
    let cfg = small_config();
    let store = Arc::new(MockGeneStore::empty());
    let sim = Arc::new(MockEvaluator::new((0..cfg.pop_size).map(|i| i as f64).collect()));
    let tier_cfg = TierConfig::default();

    let multi_tier = MultiTierEvaluator::new(
        sim,
        store.clone(),
        None,
        None,
        tier_cfg,
    );

    let mut rng = seeded_rng(&cfg);
    let runner = GaRunner {
        cfg: &cfg,
        evaluator: &multi_tier,
        store: store.as_ref(),
        neighbor_store: None,
        migration: None,
    };
    runner.run(&mut rng).await.unwrap();
    // At minimum, generation 1 evaluates initial population uncached and stores them.
    assert!(store.store_calls.load(std::sync::atomic::Ordering::SeqCst) >= cfg.pop_size);
}

#[tokio::test]
async fn runner_tracks_and_returns_best_candidate_genome() {
    let cfg = small_config();
    let mut rng = seeded_rng(&cfg);
    let initial_pop = orchestrator::ga::operators::random_population(&mut rng, &cfg);
    let expected_best_genome = initial_pop[0].clone();

    // Assign highest fitness to index 0
    let mut fitnesses = vec![10.0; cfg.pop_size];
    fitnesses[0] = 999.0;
    let evaluator = MockEvaluator::new(fitnesses);
    let store = MockGeneStore::empty();

    let mut runner_rng = seeded_rng(&cfg);
    let runner = GaRunner {
        cfg: &cfg,
        evaluator: &evaluator,
        store: &store,
        neighbor_store: None,
        migration: None,
    };
    let result = runner.run(&mut runner_rng).await.unwrap();
    assert!((result.best_fitness - 999.0).abs() < 1e-9);
    assert_eq!(result.best_genome, expected_best_genome);
}
