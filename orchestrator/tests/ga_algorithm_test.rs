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
    let runner = GaRunner::new(&cfg, &evaluator, &store);
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
    let runner = GaRunner::new(&cfg, &evaluator, &store);
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

    let mut rng = seeded_rng(&cfg);
    let runner = GaRunner::new(&cfg, &multi_tier, store.as_ref());
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
    let runner = GaRunner::new(&cfg, &evaluator, &store);
    runner.run(&mut rng).await.unwrap();
    assert_eq!(
        store.evict_calls.load(std::sync::atomic::Ordering::SeqCst),
        cfg.max_generations
    );
}

#[tokio::test]
async fn runner_stops_early_when_stagnated() {
    let mut cfg = small_config();
    cfg.max_generations = 20;
    cfg.min_generations = 2;
    cfg.stagnation_patience = 3;
    cfg.min_improvement = 0.01;

    let pop = cfg.pop_size;
    let fitnesses = vec![5.0; pop];
    let evaluator = MockEvaluator::new(fitnesses);
    let store = MockGeneStore::empty();

    let mut rng = seeded_rng(&cfg);
    let runner = GaRunner::new(&cfg, &evaluator, &store);
    let res = runner.run(&mut rng).await.unwrap();

    assert_eq!(res.best_fitness, 5.0);
    let evictions = store.evict_calls.load(std::sync::atomic::Ordering::SeqCst);
    // Gen 1: baseline (stagnant = 0)
    // Gen 2: stagnant = 1
    // Gen 3: stagnant = 2
    // Gen 4: stagnant = 3 >= patience (3) -> stops at gen 4
    assert_eq!(evictions, 4);
    assert!(evictions < cfg.max_generations);
}

#[tokio::test]
async fn runner_continues_while_making_progress() {
    let mut cfg = small_config();
    cfg.max_generations = 5;
    cfg.min_generations = 1;
    cfg.stagnation_patience = 2;
    cfg.min_improvement = 0.1;

    struct DynamicEvaluator {
        counter: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl orchestrator::evaluator::Evaluator for DynamicEvaluator {
        async fn evaluate_population(&self, pop: &[Vec<f64>]) -> Result<Vec<f64>, tonic::Status> {
            let step = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let val = 10.0 + (step as f64) * 5.0;
            Ok(vec![val; pop.len()])
        }
    }

    let evaluator = DynamicEvaluator {
        counter: std::sync::atomic::AtomicUsize::new(0),
    };
    let store = MockGeneStore::empty();

    let mut rng = seeded_rng(&cfg);
    let runner = GaRunner::new(&cfg, &evaluator, &store);
    let res = runner.run(&mut rng).await.unwrap();
    assert_eq!(
        store.evict_calls.load(std::sync::atomic::Ordering::SeqCst),
        cfg.max_generations
    );
    assert_eq!(res.best_fitness, 30.0);
}

#[test]
fn progress_tracker_detects_stagnation_after_patience() {
    use orchestrator::ga::algorithm::ProgressTracker;

    let mut tracker = ProgressTracker::new(3, 0.01, 2);
    assert_eq!(tracker.stagnant_generations(), 0);

    // Gen 1: initial baseline
    assert!(!tracker.update(10.0, 1));
    assert_eq!(tracker.stagnant_generations(), 0);

    // Gen 2: stagnant (10.0 <= 10.0 + 0.01)
    assert!(!tracker.update(10.0, 2));
    assert_eq!(tracker.stagnant_generations(), 1);

    // Gen 3: stagnant (sub-threshold improvement)
    assert!(!tracker.update(10.005, 3));
    assert_eq!(tracker.stagnant_generations(), 2);

    // Gen 4: reaches patience = 3, gen 4 >= min_gen 2 -> converged!
    assert!(tracker.update(10.008, 4));
    assert_eq!(tracker.stagnant_generations(), 3);
    assert!(tracker.is_stagnant(4));
}

#[test]
fn progress_tracker_resets_on_significant_improvement() {
    use orchestrator::ga::algorithm::ProgressTracker;

    let mut tracker = ProgressTracker::new(3, 0.01, 1);
    tracker.update(10.0, 1);
    tracker.update(10.0, 2);
    assert_eq!(tracker.stagnant_generations(), 1);

    // Significant progress (> 0.01) resets counter
    assert!(!tracker.update(10.05, 3));
    assert_eq!(tracker.stagnant_generations(), 0);
    assert_eq!(tracker.last_improvement_fitness(), 10.05);
}

#[test]
fn progress_tracker_disabled_when_patience_is_zero() {
    use orchestrator::ga::algorithm::ProgressTracker;

    let mut tracker = ProgressTracker::new(0, 0.01, 1);
    tracker.update(10.0, 1);
    assert!(!tracker.update(10.0, 2));
    assert!(!tracker.update(10.0, 3));
    assert!(!tracker.is_stagnant(100));
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
    let runner = GaRunner::new(&cfg, &multi_tier, store.as_ref());
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
    let runner = GaRunner::new(&cfg, &evaluator, &store);
    let result = runner.run(&mut runner_rng).await.unwrap();
    assert!((result.best_fitness - 999.0).abs() < 1e-9);
    assert_eq!(result.best_genome, expected_best_genome);
}

