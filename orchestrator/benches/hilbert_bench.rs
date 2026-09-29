//! Benchmark A: Multi-Probe Hilbert Curve Encoding & Key Generation
//!
//! # Objective
//! Measure and optimize CPU execution time, memory allocation patterns, and
//! throughput for:
//! 1. Raw Hilbert coordinate transformation (`hilbert_encode_hex`)
//! 2. Multi-probe rotated key generation across 3 curves (`keys_for`)
//! 3. Key splitting and payload deserialization (`parse_key`)

#![allow(unused_variables, dead_code, unused_imports, unused_mut)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use orchestrator::hilbert::{GeneKeyGenerator, HilbertEncoder, HilbertKeyGenerator, BITS};

/// 1. Benchmark raw single-curve Hilbert encoding for a 10D normalized point.
fn bench_hilbert_encode_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("hilbert_encode_single");

    let point = vec![0.12, 0.45, 0.78, 0.23, 0.89, 0.34, 0.56, 0.67, 0.91, 0.05];
    let encoder = HilbertEncoder::new(10, BITS);
    let hex_width = (10 * BITS).div_ceil(4);

    group.bench_function("encode_hex_10d", |b| {
        b.iter(|| encoder.encode_hex(black_box(&point), black_box(0), black_box(hex_width)));
    });

    group.finish();
}

/// 2. Benchmark full multi-probe key generation (`GeneKeyGenerator::keys_for`).
fn bench_multi_probe_key_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_probe_key_generation");

    let gene_lengths = [5, 10, 15];

    for &len in &gene_lengths {
        let keygen = HilbertKeyGenerator::new(len);
        let genes: Vec<f64> = (0..len)
            .map(|i| (i as f64 + 1.0) / (len as f64 + 1.0))
            .collect();

        group.bench_with_input(BenchmarkId::new("keys_for", len), &genes, |b, g| {
            b.iter(|| keygen.keys_for(black_box(g)));
        });
    }

    group.finish();
}

/// 3. Benchmark DHT key parsing (`GeneKeyGenerator::parse_key`).
fn bench_key_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("hilbert_key_parsing");

    let keygen = HilbertKeyGenerator::new(10);
    let sample_key = "01a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4|[0.12,0.45,0.78,0.23,0.89,0.34,0.56,0.67,0.91,0.05]";

    group.bench_function("parse_key_10d", |b| {
        b.iter(|| keygen.parse_key(black_box(sample_key)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hilbert_encode_single,
    bench_multi_probe_key_generation,
    bench_key_parsing
);
criterion_main!(benches);
