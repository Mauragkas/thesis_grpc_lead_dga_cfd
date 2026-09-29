//! Benchmark D: In-Memory Gene Store, KNN & Distance Metrics
//!
//! # Objective
//! Measure and profile nearest-neighbor queries, exact caching, and eviction:
//! 1. Raw 10D Euclidean distance metric calculation (`EuclideanDistance::distance`)
//! 2. In-memory KNN linear scan (`InMemoryGeneStore::query_knn`) across population sizes
//! 3. Exact gene cache lookup (`InMemoryGeneStore::lookup_exact`)
//! 4. Generation-based TTL eviction (`InMemoryGeneStore::evict_expired`)
//!
//! # Why this matters
//! - The gene store caches evaluated individuals to prevent redundant aerodynamic simulations.
//! - KNN search scales as $O(N \cdot d)$ per query.
//! - Benchmarking guides decisions on SIMD vectorization for distance math, or upgrading
//!   from brute-force scan to spatial structures (e.g. HNSW or VP-Tree).

#![allow(unused_variables, dead_code, unused_imports, unused_mut)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use orchestrator::gene_store::{
    DistanceMetric, EuclideanDistance, GeneStore, GenerationEvictor, InMemoryGeneStore,
};
use tokio::runtime::Runtime;

/// 1. Benchmark pure 10D Euclidean distance calculation.
///
/// Distance calculation is the core inner loop of all KNN searches.
/// Measures baseline compiler auto-vectorization vs potential AVX2/NEON SIMD optimizations.
fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metric");

    let metric = EuclideanDistance;
    let a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let b = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];

    group.bench_function("euclidean_10d", |bench| {
        bench.iter(|| metric.distance(black_box(&a), black_box(&b)));
    });

    group.finish();
}

/// 2. Benchmark In-Memory KNN query across store sizes.
///
/// Measures the scaling behavior of brute-force scan + distance computation + sorting.
/// Pinpoints the exact threshold where an index-backed store becomes necessary.
fn bench_gene_store_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("gene_store_knn");
    let rt = Runtime::new().unwrap();

    let store_sizes = [100, 1000, 5000];
    let k = 10;
    let query_genes = [0.5; 10];

    for &size in &store_sizes {
        let store = InMemoryGeneStore::new(EuclideanDistance, GenerationEvictor { max_age: 50 });

        // Pre-populate store with synthetic individuals
        rt.block_on(async {
            for i in 0..size {
                let genes: Vec<f64> = (0..10)
                    .map(|d| ((i * 17 + d * 31) % 100) as f64 / 100.0)
                    .collect();
                store.store(genes, 100.0 - (i as f64 * 0.01), 1).await;
            }
        });

        group.bench_with_input(BenchmarkId::new("records", size), &size, |bench, _| {
            bench.iter(|| {
                rt.block_on(async {
                    store
                        .query_knn(black_box(&query_genes), black_box(k), black_box(1))
                        .await
                })
            });
        });
    }

    group.finish();
}

/// 3. Benchmark exact match lookup (cache hit vs cache miss).
///
/// Exact lookup runs before CFD/evaluation dispatch to avoid re-evaluating known geometries.
/// Determines whether a hash-map index should augment the linear record scan for exact matching.
fn bench_exact_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("gene_store_exact_lookup");
    let rt = Runtime::new().unwrap();

    let store = InMemoryGeneStore::new(EuclideanDistance, GenerationEvictor { max_age: 50 });
    let hit_genes = vec![0.25; 10];
    let miss_genes = [0.99; 10];

    rt.block_on(async {
        for i in 0..500 {
            let genes = if i == 250 {
                hit_genes.clone()
            } else {
                vec![(i as f64) / 500.0; 10]
            };
            store.store(genes, 50.0, 1).await;
        }
    });

    group.bench_function("lookup_hit_500_records", |bench| {
        bench.iter(|| {
            rt.block_on(async {
                store
                    .lookup_exact(black_box(&hit_genes), black_box(1))
                    .await
            })
        });
    });

    group.bench_function("lookup_miss_500_records", |bench| {
        bench.iter(|| {
            rt.block_on(async {
                store
                    .lookup_exact(black_box(&miss_genes), black_box(1))
                    .await
            })
        });
    });

    group.finish();
}

/// 4. Benchmark record eviction under TTL / stale generation policy.
///
/// Measures `retain` predicate scanning and memory deallocation cost during periodic housekeeping.
fn bench_eviction_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("gene_store_eviction");
    let rt = Runtime::new().unwrap();

    group.bench_function("evict_expired_1000_records", |bench| {
        bench.iter_batched(
            || {
                let store =
                    InMemoryGeneStore::new(EuclideanDistance, GenerationEvictor { max_age: 10 });
                rt.block_on(async {
                    for i in 0..1000 {
                        let gen = if i % 2 == 0 { 5 } else { 15 };
                        store.store(vec![0.5; 10], 50.0, gen).await;
                    }
                });
                store
            },
            |store| {
                rt.block_on(async {
                    store.evict_expired(black_box(20)).await;
                })
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_euclidean_distance,
    bench_gene_store_knn,
    bench_exact_lookup,
    bench_eviction_scan
);
criterion_main!(benches);
