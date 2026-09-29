//! Benchmark C: RMI Training & Federated Averaging
//!
//! # Objective
//! Measure and profile background training and cluster synchronization routines:
//! 1. Standard linear leaf model training (`RmiModel::train`)
//! 2. Mountain-climbing auto-configuration (`RmiModel::train_auto`)
//! 3. Federated model averaging across nodes (`fed_avg`)
//! 4. 2-bit online PID anchor adjustment (`PidTuner::adjust`)
//!
//! # Why this matters
//! - LEAD retrains models asynchronously when key distributions drift.
//! - `train_auto` performs iterative model evaluation on a 1% sketch.
//! - Federated averaging merges models across multiple ring nodes during coordinator rounds.
//! - Fast retraining keeps node CPU load low during heavy insert traffic.

#![allow(unused_variables, dead_code, unused_imports, unused_mut)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lead_node::lead::learning::{fed_avg, PidTuner};
use lead_node::rmi::{Anchor, LeafKind, LinearLeaf, PidState, RmiModel};

/// Helper: generate synthetic hex keys for training benchmarks.
fn generate_keys(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            let prefix = format!("{:016x}", (i as u64).wrapping_mul(0x9e3779b97f4a7c15));
            format!("{prefix}|[0.1,0.2,0.3]")
        })
        .collect()
}

/// 1. Benchmark basic linear RMI model training.
///
/// Measures time spent in key sorting, bucket partitioning, and linear regression (`linreg`).
fn bench_train_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmi_train_linear");

    let dataset_sizes = [500, 2000, 10000];

    for &size in &dataset_sizes {
        let keys = generate_keys(size);

        group.bench_with_input(BenchmarkId::new("keys", size), &keys, |b, k| {
            b.iter(|| RmiModel::train(black_box(k), black_box(1)));
        });
    }

    group.finish();
}

/// 2. Benchmark auto-model training with mountain-climbing optimization.
///
/// `train_auto` builds a 1% sketch, tests multiple bin counts (8, 16, 32, 64),
/// and evaluates RadixSpline vs Linear error metrics.
fn bench_train_auto(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmi_train_auto");

    let dataset_sizes = [1000, 5000];

    for &size in &dataset_sizes {
        let keys = generate_keys(size);

        group.bench_with_input(BenchmarkId::new("auto_keys", size), &keys, |b, k| {
            b.iter(|| RmiModel::train_auto(black_box(k), black_box(1)));
        });
    }

    group.finish();
}

/// 3. Benchmark Federated Averaging (`fed_avg`) across peer models.
///
/// During transient coordinator rounds, one node aggregates models from all peers.
/// Measures weighted leaf parameter blending and anchor offset synchronization overhead.
fn bench_federated_averaging(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmi_fed_avg");

    let node_counts = [3, 8, 16];
    let bins = 16;

    for &num_nodes in &node_counts {
        let models: Vec<RmiModel> = (0..num_nodes)
            .map(|i| RmiModel {
                stage0_bins: bins,
                leaves: vec![
                    LeafKind::Linear(LinearLeaf {
                        weight: 1.0 + (i as f64 * 0.05),
                        bias: i as f64 * 0.01,
                        anchor: Anchor {
                            offset: (i as f64) * 0.001,
                            scale: 1.0,
                        },
                    });
                    bins
                ],
                n: 1000 * (i + 1),
                version: 1,
                pid_state: vec![PidState::default(); bins],
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("nodes", num_nodes), &models, |b, m| {
            b.iter(|| fed_avg(black_box(m.clone()), black_box(2)));
        });
    }

    group.finish();
}

/// 4. Benchmark 2-bit online PID controller adjustments.
///
/// Evaluates the per-leaf anchor adjustment loop run on every storage update or insertion.
fn bench_pid_tuner_adjustment(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmi_pid_tuner");

    let tuner = PidTuner::default();
    let mut state = PidState::default();
    let mut anchor = Anchor::default();

    group.bench_function("pid_adjust_single_leaf", |b| {
        b.iter(|| {
            tuner.adjust(
                black_box(&mut state),
                black_box(&mut anchor),
                black_box(95),
                black_box(5),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_train_linear,
    bench_train_auto,
    bench_federated_averaging,
    bench_pid_tuner_adjustment
);
criterion_main!(benches);
