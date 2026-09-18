//! Tests for `config` module: defaults, env overrides, and the `clip` helper.

mod common;

use orchestrator::config::{
    clip, config_from_vars, GaConfig, GeneStoreConfig, TierConfig, TransportConfig,
};
use std::time::Duration;

#[test]
fn defaults_are_sane() {
    let ga = GaConfig::default();
    assert!(ga.pop_size > 0);
    assert!(ga.genes_len > 0);
    assert!(ga.max_generations > 0);
    assert!(ga.generations() > 0);
    assert!(ga.min_generations > 0);
    assert!(ga.stagnation_patience > 0);
    assert!(ga.min_improvement > 0.0);
    assert!(ga.elite_frac > 0.0 && ga.elite_frac <= 1.0);
    assert!(ga.batch_size > 0);
    assert!(!ga.eval_endpoint.is_empty());

    let t = TransportConfig::default();
    assert!(t.rpc_timeout > Duration::ZERO);
    assert!(t.max_attempts > 0);

    let s = GeneStoreConfig::default();
    assert!(s.max_age_generations > 0);

    let tier = TierConfig::default();
    assert!(tier.epsilon_exact > 0.0);
    assert!(tier.radius_r > tier.epsilon_exact);
    assert!(tier.k_neighbors > 0);
}

#[test]
fn clip_bounds_value_to_unit_interval() {
    assert_eq!(clip(-5.0), 0.0);
    assert_eq!(clip(0.0), 0.0);
    assert_eq!(clip(0.5), 0.5);
    assert_eq!(clip(1.0), 1.0);
    assert_eq!(clip(5.0), 1.0);
}

#[test]
fn clip_is_idempotent_at_bounds() {
    assert_eq!(clip(clip(10.0)), 1.0);
    assert_eq!(clip(clip(-10.0)), 0.0);
}

#[test]
fn config_from_vars_uses_defaults_when_empty() {
    let empty_vars: [(&str, &str); 0] = [];
    let (ga, _t, store, lead, surrogate, tier, ring, migration) = config_from_vars(empty_vars);
    assert_eq!(ga.eval_endpoint, "load-balancer:50051");
    assert_eq!(ga.seed, 42);
    assert_eq!(ga.max_generations, 100);
    assert_eq!(ga.min_generations, 10);
    assert_eq!(ga.stagnation_patience, 10);
    assert_eq!(ga.min_improvement, 0.001);
    assert_eq!(store.max_age_generations, 5);
    assert!(lead.endpoint.is_none());
    assert!(surrogate.endpoint.is_none());
    assert_eq!(tier.epsilon_exact, 0.005);
    assert_eq!(tier.radius_r, 0.15);
    assert_eq!(tier.k_neighbors, 15);
    assert_eq!(ring.bind_address, "0.0.0.0:50060");
    assert_eq!(ring.self_address, "orchestrator:50060");
    assert!(ring.bootstrap_address.is_none());
    assert_eq!(migration.interval_generations, 5);
    assert_eq!(migration.migrant_count, 3);
}

#[test]
fn config_from_vars_reads_overrides() {
    let vars = [
        ("EVAL_ENDPOINT", "worker:9999"),
        ("GA_SEED", "123"),
        ("MAX_GENERATIONS", "50"),
        ("MIN_GENERATIONS", "5"),
        ("STAGNATION_PATIENCE", "8"),
        ("MIN_IMPROVEMENT", "0.005"),
        ("GENE_STORE_MAX_AGE", "17"),
        ("LEAD_ENDPOINT", "lead-1:2001"),
        ("SURROGATE_ENDPOINT", "surrogate-1:50054"),
        ("TIER_EPSILON_EXACT", "0.01"),
        ("TIER_RADIUS_R", "0.25"),
        ("TIER_K_NEIGHBORS", "20"),
        ("RING_BIND", "0.0.0.0:60000"),
        ("RING_SELF_ADDRESS", "orch-1:60000"),
        ("RING_BOOTSTRAP", "orch-0:60000"),
        ("MIGRATION_INTERVAL", "10"),
        ("MIGRATION_COUNT", "4"),
    ];

    let (ga, _t, store, lead, surrogate, tier, ring, migration) = config_from_vars(vars);
    assert_eq!(ga.eval_endpoint, "worker:9999");
    assert_eq!(ga.seed, 123);
    assert_eq!(ga.max_generations, 50);
    assert_eq!(ga.min_generations, 5);
    assert_eq!(ga.stagnation_patience, 8);
    assert_eq!(ga.min_improvement, 0.005);
    assert_eq!(store.max_age_generations, 17);
    assert_eq!(lead.endpoint.as_deref(), Some("lead-1:2001"));
    assert_eq!(surrogate.endpoint.as_deref(), Some("surrogate-1:50054"));
    assert_eq!(tier.epsilon_exact, 0.01);
    assert_eq!(tier.radius_r, 0.25);
    assert_eq!(tier.k_neighbors, 20);
    assert_eq!(ring.bind_address, "0.0.0.0:60000");
    assert_eq!(ring.self_address, "orch-1:60000");
    assert_eq!(ring.bootstrap_address.as_deref(), Some("orch-0:60000"));
    assert_eq!(migration.interval_generations, 10);
    assert_eq!(migration.migrant_count, 4);
}

#[test]
fn config_from_vars_reads_alternative_generation_env_names() {
    let vars = [
        ("GA_GENERATIONS", "75"),
        ("GA_MIN_GENERATIONS", "12"),
        ("GA_STAGNATION_PATIENCE", "15"),
        ("GA_MIN_IMPROVEMENT", "0.0002"),
    ];
    let (ga, _, _, _, _, _, _, _) = config_from_vars(vars);
    assert_eq!(ga.max_generations, 75);
    assert_eq!(ga.min_generations, 12);
    assert_eq!(ga.stagnation_patience, 15);
    assert_eq!(ga.min_improvement, 0.0002);
}

#[test]
fn config_from_vars_ignores_invalid_numerical_values() {
    let vars = [
        ("GA_SEED", "not-a-number"),
        ("MAX_GENERATIONS", "not-a-number"),
        ("MIN_GENERATIONS", "bad"),
        ("STAGNATION_PATIENCE", "invalid"),
        ("MIN_IMPROVEMENT", "bad-float"),
        ("GENE_STORE_MAX_AGE", "invalid"),
        ("MIGRATION_INTERVAL", "bad"),
        ("MIGRATION_COUNT", "also-bad"),
    ];
    let (ga, _, store, _, _, _, _, migration) = config_from_vars(vars);
    assert_eq!(ga.seed, 42); // falls back to default
    assert_eq!(ga.max_generations, 100);
    assert_eq!(ga.min_generations, 10);
    assert_eq!(ga.stagnation_patience, 10);
    assert_eq!(ga.min_improvement, 0.001);
    assert_eq!(store.max_age_generations, 5);
    assert_eq!(migration.interval_generations, 5);
    assert_eq!(migration.migrant_count, 3);
}
