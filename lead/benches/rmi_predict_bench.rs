//! Benchmark B: Recursive Model Index (RMI) Inference, Feature Extraction & B-Tree Comparison
//!
//! # Objective
//! Measure latency and CPU throughput of the learned indexing inference path:
//! 1. Feature extraction from key strings (`feature(key)`)
//! 2. 2-stage RMI model prediction with linear leaf models (`model.predict(key)`)
//! 3. 2-stage RMI model prediction with radix spline leaf models (`model.predict(key)`)
//! 4. Direct head-to-head comparison: RmiModel::predict vs std::collections::BTreeMap::get

#![allow(unused_variables, dead_code, unused_imports, unused_mut)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lead_node::rmi::{
    feature, Anchor, LeafKind, LinearLeaf, RadixSplineLeaf, RmiModel, RADIX_ENTRIES, RP,
};
use std::collections::BTreeMap;

/// 1. Benchmark feature extraction for different key formats.
fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmi_feature_extraction");

    let hilbert_key = "01a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4|[0.12,0.45,0.78,0.23,0.89]";
    let fallback_key = "plain_string_key_without_hex_prefix_1234567890";

    group.bench_function("hilbert_prefix_feature", |b| {
        b.iter(|| feature(black_box(hilbert_key)));
    });

    group.bench_function("fallback_hash_feature", |b| {
        b.iter(|| feature(black_box(fallback_key)));
    });

    group.finish();
}

/// 2. Benchmark RMI prediction with Linear Leaf models across Stage-0 bin counts.
fn bench_linear_leaf_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmi_predict_linear");

    let bin_counts = [8, 16, 64, 256];
    let key = "01a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4|[0.12,0.45,0.78]";

    for &bins in &bin_counts {
        let model = RmiModel {
            stage0_bins: bins,
            leaves: vec![
                LeafKind::Linear(LinearLeaf {
                    weight: 1.25,
                    bias: -0.05,
                    anchor: Anchor::default(),
                });
                bins
            ],
            n: 1000,
            version: 1,
            pid_state: vec![],
        };

        group.bench_with_input(BenchmarkId::new("bins", bins), &model, |b, m| {
            b.iter(|| m.predict(black_box(key)));
        });
    }

    group.finish();
}

/// 3. Benchmark RMI prediction with Radix Spline Leaf models.
fn bench_radix_spline_leaf_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmi_predict_radix_spline");

    let bins = 16;
    let table = vec![1000u32; RADIX_ENTRIES];
    let model = RmiModel {
        stage0_bins: bins,
        leaves: vec![
            LeafKind::RadixSpline(RadixSplineLeaf {
                radix_table: table,
                rp: RP,
                anchor: Anchor::default(),
            });
            bins
        ],
        n: 5000,
        version: 1,
        pid_state: vec![],
    };

    let key = "01a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4|[0.5,0.5,0.5]";

    group.bench_function("radix_spline_16bins", |b| {
        b.iter(|| model.predict(black_box(key)));
    });

    group.finish();
}

/// 4. Direct head-to-head comparison: RMI Inference vs std::collections::BTreeMap lookup.
fn bench_rmi_vs_btree_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_lookup_rmi_vs_btree");

    let dataset_sizes = [1_000, 10_000, 50_000];

    for &size in &dataset_sizes {
        // Generate keys
        let keys: Vec<String> = (0..size)
            .map(|i| {
                let prefix = format!("{:016x}", (i as u64).wrapping_mul(0x9e3779b97f4a7c15));
                format!("{prefix}|[0.1,0.2,0.3]")
            })
            .collect();

        // 1. Train real RmiModel on keys
        let rmi = RmiModel::train(&keys, 1);

        // 2. Build BTreeMap
        let mut btree = BTreeMap::new();
        for (idx, k) in keys.iter().enumerate() {
            btree.insert(k.clone(), idx as u64);
        }

        // Test probe key
        let query_key = &keys[size / 2];

        group.bench_with_input(BenchmarkId::new("RMI_Linear", size), &rmi, |b, model| {
            b.iter(|| model.predict(black_box(query_key)));
        });

        group.bench_with_input(BenchmarkId::new("std_BTreeMap", size), &btree, |b, map| {
            b.iter(|| map.get(black_box(query_key)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_feature_extraction,
    bench_linear_leaf_predict,
    bench_radix_spline_leaf_predict,
    bench_rmi_vs_btree_scaling
);
criterion_main!(benches);
