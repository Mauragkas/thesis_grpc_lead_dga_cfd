//! Benchmark E: Genetic Algorithm Breeding & Selection Operators
//!
//! # Objective
//! Measure CPU throughput of core in-memory genetic algorithm operators:
//! 1. Initial random population sampling (`random_population`)
//! 2. (μ + λ) survivor selection and fitness ranking (`select_survivors`)
//! 3. Gaussian offspring mutation and range clipping (`next_generation`)
//!
//! # Why this matters
//! - GA generation loops execute repeatedly on orchestrator nodes across hundreds of generations.
//! - Quantifies RNG overhead (`rand::Rng`, `rand_distr::Normal`), sorting bottlenecks in selection,
//!   and vector allocation overhead during offspring creation.

#![allow(unused_variables, dead_code, unused_imports, unused_mut)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use orchestrator::config::GaConfig;
use orchestrator::ga::operators::{next_generation, random_population, select_survivors};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Normal;

/// 1. Benchmark random population generation.
///
/// Evaluates uniform float RNG generation and nested vector allocation speed.
fn bench_random_population_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_random_population");

    let pop_sizes = [50, 200, 1000];

    for &pop_size in &pop_sizes {
        let cfg = GaConfig {
            pop_size,
            genes_len: 10,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(42);

        group.bench_with_input(BenchmarkId::new("pop_size", pop_size), &cfg, |b, c| {
            b.iter(|| random_population(&mut rng, black_box(c)));
        });
    }

    group.finish();
}

/// 2. Benchmark elitist survivor selection.
///
/// `select_survivors` indexes, sorts by fitness float values, and extracts top `elite_frac`.
/// Benchmarking isolates sorting cost and vector cloning overhead.
fn bench_survivor_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_select_survivors");

    let pop_sizes = [100, 500, 2000];

    for &pop_size in &pop_sizes {
        let cfg = GaConfig {
            pop_size,
            elite_frac: 0.5,
            ..Default::default()
        };

        let population: Vec<Vec<f64>> = (0..pop_size).map(|_| vec![0.5; 10]).collect();
        let fitnesses: Vec<f64> = (0..pop_size)
            .map(|i| ((i * 37) % pop_size) as f64)
            .collect();

        group.bench_with_input(BenchmarkId::new("pop_size", pop_size), &pop_size, |b, _| {
            b.iter(|| {
                select_survivors(
                    black_box(&population),
                    black_box(&fitnesses),
                    black_box(&cfg),
                )
            });
        });
    }

    group.finish();
}

/// 3. Benchmark offspring generation with Gaussian mutation and clipping.
///
/// Measures Gaussian distribution sampling (`normal.sample(&mut rng)`), element-wise addition,
/// and floating-point clamping (`clip`) in the evolutionary loop.
fn bench_offspring_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_next_generation");

    let pop_size = 100;
    let cfg = GaConfig {
        pop_size,
        mut_sigma: 0.08,
        ..Default::default()
    };

    let survivors: Vec<Vec<f64>> = (0..50).map(|_| vec![0.5; 10]).collect();
    let normal = Normal::new(0.0, cfg.mut_sigma).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    group.bench_function("mutate_and_clip_100_children", |b| {
        b.iter(|| {
            next_generation(
                black_box(&survivors),
                black_box(&cfg),
                &mut rng,
                black_box(&normal),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_random_population_generation,
    bench_survivor_selection,
    bench_offspring_generation
);
criterion_main!(benches);
