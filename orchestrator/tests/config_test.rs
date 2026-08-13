//! Tests for `config` module: defaults, env overrides, and the `clip` helper.

mod common;

use orchestrator::config::{clip, config_from_env, GaConfig, GeneStoreConfig, TransportConfig};
use serial_test::serial;
use std::time::Duration;

#[test]
fn defaults_are_sane() {
    let ga = GaConfig::default();
    assert!(ga.pop_size > 0);
    assert!(ga.genes_len > 0);
    assert!(ga.generations > 0);
    assert!(ga.elite_frac > 0.0 && ga.elite_frac <= 1.0);
    assert!(ga.batch_size > 0);
    assert!(!ga.eval_endpoint.is_empty());

    let t = TransportConfig::default();
    assert!(t.rpc_timeout > Duration::ZERO);
    assert!(t.max_attempts > 0);

    let s = GeneStoreConfig::default();
    assert!(s.max_age_generations > 0);
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
#[serial]
fn config_from_env_uses_defaults_when_unset() {
    std::env::remove_var("EVAL_ENDPOINT");
    std::env::remove_var("GA_SEED");
    std::env::remove_var("GENE_STORE_MAX_AGE");

    let (ga, _t, store, _lead, _ring, _migration) = config_from_env();
    assert_eq!(ga.eval_endpoint, "load-balancer:50051");
    assert_eq!(ga.seed, 42);
    assert_eq!(store.max_age_generations, 5);
}

#[test]
#[serial]
fn config_from_env_reads_overrides() {
    std::env::set_var("EVAL_ENDPOINT", "worker:9999");
    std::env::set_var("GA_SEED", "123");
    std::env::set_var("GENE_STORE_MAX_AGE", "17");

    let (ga, _t, store, _lead, _ring, _migration) = config_from_env();
    assert_eq!(ga.eval_endpoint, "worker:9999");
    assert_eq!(ga.seed, 123);
    assert_eq!(store.max_age_generations, 17);

    // cleanup
    std::env::remove_var("EVAL_ENDPOINT");
    std::env::remove_var("GA_SEED");
    std::env::remove_var("GENE_STORE_MAX_AGE");
}

#[test]
#[serial]
fn config_from_env_ignores_invalid_seed() {
    std::env::set_var("GA_SEED", "not-a-number");
    let (ga, _, _, _, _, _) = config_from_env();
    assert_eq!(ga.seed, 42); // falls back to default
    std::env::remove_var("GA_SEED");
}
