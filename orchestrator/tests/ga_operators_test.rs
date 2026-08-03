//! Tests for `ga::operators`: population generation, survivor selection,
//! and next-generation breeding. These are pure functions — no I/O.

mod common;

use orchestrator::config::GaConfig;
use orchestrator::ga::operators::{next_generation, random_population, select_survivors};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Normal;

#[test]
fn random_population_has_correct_shape() {
    let cfg = common::small_config();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let pop = random_population(&mut rng, &cfg);
    assert_eq!(pop.len(), cfg.pop_size);
    for ind in &pop {
        assert_eq!(ind.len(), cfg.genes_len);
        for g in ind {
            assert!((*g) >= 0.0 && (*g) <= 1.0, "gene out of [0,1]: {g}");
        }
    }
}

#[test]
fn random_population_is_deterministic_with_seed() {
    let cfg = common::small_config();
    let mut a = StdRng::seed_from_u64(cfg.seed);
    let mut b = StdRng::seed_from_u64(cfg.seed);
    assert_eq!(
        random_population(&mut a, &cfg),
        random_population(&mut b, &cfg)
    );
}

#[test]
fn select_survivors_picks_top_by_fitness() {
    let cfg = common::small_config();
    let pop: Vec<Vec<f64>> = (0..cfg.pop_size).map(|i| vec![i as f64]).collect();
    let fitnesses: Vec<f64> = (0..cfg.pop_size).map(|i| i as f64).collect();
    let survivors = select_survivors(&pop, &fitnesses, &cfg);
    let n_expected = ((cfg.pop_size as f64) * cfg.elite_frac) as usize;
    assert_eq!(survivors.len(), n_expected);
    // highest-fitness individuals should be the last indices
    for s in &survivors {
        assert_eq!(s[0] as usize >= cfg.pop_size - n_expected, true);
    }
}

#[test]
fn select_survivors_never_returns_fewer_than_two() {
    let cfg = GaConfig {
        elite_frac: 0.0,
        ..common::small_config()
    };
    let pop = vec![vec![0.0], vec![1.0], vec![2.0]];
    let fitnesses = vec![1.0, 2.0, 3.0];
    let survivors = select_survivors(&pop, &fitnesses, &cfg);
    assert!(survivors.len() >= 2);
}

#[test]
fn next_generation_refills_to_pop_size() {
    let cfg = common::small_config();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let normal = Normal::new(0.0, cfg.mut_sigma).unwrap();
    let survivors = vec![vec![0.5; cfg.genes_len], vec![0.2; cfg.genes_len]];
    let next = next_generation(&survivors, &cfg, &mut rng, &normal);
    assert_eq!(next.len(), cfg.pop_size);
    // survivors are preserved at the front
    for (i, s) in survivors.iter().enumerate() {
        assert_eq!(next[i], *s);
    }
}

#[test]
fn next_generation_children_are_clipped_to_unit_interval() {
    let cfg = common::small_config();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    // large sigma => mutations likely to exceed bounds, clip must hold
    let normal = Normal::new(0.0, 10.0).unwrap();
    let survivors = vec![vec![0.0; cfg.genes_len], vec![1.0; cfg.genes_len]];
    let next = next_generation(&survivors, &cfg, &mut rng, &normal);
    for ind in &next {
        for g in ind {
            assert!((*g) >= 0.0 && (*g) <= 1.0, "child gene out of bounds: {g}");
        }
    }
}
