//! Integration tests for `GaRunner`. Uses mock `Evaluator` and `GeneStore`
//! from `common` so the algorithm is tested in isolation from gRPC/storage.

mod common;

use common::{small_config, MockEvaluator, MockGeneStore};
use orchestrator::ga::algorithm::GaRunner;
use rand::rngs::StdRng;
use rand::SeedableRng;
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
    };
    let best = runner.run(&mut rng).await.unwrap();
    // best ever should be the max of the supplied fitnesses
    assert!((best - (pop as f64)).abs() < 1e-9);
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
    };
    let err = runner.run(&mut rng).await.unwrap_err();
    assert_eq!(err.code(), Status::internal("x").code());
    assert!(err.message().contains("mock evaluator failure"));
}

#[tokio::test]
async fn runner_skips_evaluator_for_cached_individuals() {
    let cfg = small_config();
    let mut rng = seeded_rng(&cfg);
    // Build the initial population deterministically so we can seed hits.
    let pop = orchestrator::ga::operators::random_population(&mut rng, &cfg);

    // Seed the store with exact hits for the first half of the population.
    let store = MockGeneStore::empty();
    let half = pop.len() / 2;
    for g in pop.iter().take(half) {
        store.seed_exact(g, 100.0);
    }

    // Evaluator returns distinct values so we can confirm it ran.
    let evaluator = MockEvaluator::new((0..pop.len()).map(|i| i as f64).collect());

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let runner = GaRunner {
        cfg: &cfg,
        evaluator: &evaluator,
        store: &store,
    };
    let best = runner.run(&mut rng).await.unwrap();
    // At least one generation ran; cached individuals returned fitness 100.0
    assert!(best >= 100.0);
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
    };
    runner.run(&mut rng).await.unwrap();
    assert_eq!(
        store.evict_calls.load(std::sync::atomic::Ordering::SeqCst),
        cfg.generations
    );
}

#[tokio::test]
async fn runner_stores_each_newly_evaluated_individual() {
    let cfg = small_config();
    let evaluator = MockEvaluator::new((0..cfg.pop_size).map(|i| i as f64).collect());
    let store = MockGeneStore::empty();

    let mut rng = seeded_rng(&cfg);
    let runner = GaRunner {
        cfg: &cfg,
        evaluator: &evaluator,
        store: &store,
    };
    runner.run(&mut rng).await.unwrap();
    // At minimum, generation 1 evaluates the full initial population uncached.
    assert!(store.store_calls.load(std::sync::atomic::Ordering::SeqCst) >= cfg.pop_size);
}
