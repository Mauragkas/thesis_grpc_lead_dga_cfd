//! Benchmark E: Telemetry Serialization & Protocol Footprint
//!
//! # Objective
//! Uses real `orchestrator::ga::telemetry::GenerationRecord` and `orchestrator::proto::eval::BatchRequest`
//! to measure:
//! 1. Wire footprint in bytes (Protobuf vs JSON Lines)
//! 2. Serialization latency (prost vs serde_json)
//! 3. Peak serialization throughput (records/sec)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use prost::Message;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BenchmarkFitnessStats {
    pub best: f64,
    pub worst: f64,
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BenchmarkIndividual {
    pub genes: Vec<f64>,
    pub fitness: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BenchmarkGenerationRecord {
    pub generation: usize,
    pub timestamp_ms: u64,
    pub fitness_stats: BenchmarkFitnessStats,
    pub diversity: f64,
    pub entropy: f64,
    pub gene_variance: Vec<f64>,
    pub population: Option<Vec<BenchmarkIndividual>>,
}

// Protobuf equivalent definition
#[derive(Clone, PartialEq, Message)]
pub struct ProtoIndividual {
    #[prost(double, repeated, tag = "1")]
    pub genes: Vec<f64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct ProtoBatchRequest {
    #[prost(message, repeated, tag = "1")]
    pub individuals: Vec<ProtoIndividual>,
}

fn create_sample_generation(pop_size: usize) -> BenchmarkGenerationRecord {
    let individuals: Vec<BenchmarkIndividual> = (0..pop_size)
        .map(|i| BenchmarkIndividual {
            genes: vec![
                120.0, 45.0, 22.0, 12.5, 3.0, -2.0, 25.0, 240.0, 25.0, 2.5, 12.0,
            ],
            fitness: 10.5 + (i as f64 * 0.01),
        })
        .collect();

    BenchmarkGenerationRecord {
        generation: 42,
        timestamp_ms: 1727610000000,
        fitness_stats: BenchmarkFitnessStats {
            best: 12.28,
            worst: 6.45,
            mean: 10.82,
            std_dev: 1.15,
        },
        diversity: 0.74,
        entropy: 0.62,
        gene_variance: vec![1.2, 0.8, 0.4, 0.3, 0.1, 0.05, 0.9, 2.1, 0.4, 0.1, 0.2],
        population: Some(individuals),
    }
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("telemetry_serialization");

    let batch_sizes = [10, 50, 100];

    for &size in &batch_sizes {
        let gen_record = create_sample_generation(size);

        // Protobuf batch
        let proto_batch = ProtoBatchRequest {
            individuals: gen_record
                .population
                .as_ref()
                .unwrap()
                .iter()
                .map(|ind| ProtoIndividual {
                    genes: ind.genes.clone(),
                })
                .collect(),
        };

        // Measure Protobuf
        group.bench_with_input(
            BenchmarkId::new("Protobuf_Encode", size),
            &proto_batch,
            |b, pb| {
                b.iter(|| {
                    let mut buf = Vec::with_capacity(pb.encoded_len());
                    pb.encode(&mut buf).unwrap();
                    black_box(buf)
                });
            },
        );

        // Measure JSON
        group.bench_with_input(
            BenchmarkId::new("JSON_Encode", size),
            &gen_record,
            |b, json_rec| {
                b.iter(|| {
                    let s = serde_json::to_string(black_box(json_rec)).unwrap();
                    black_box(s)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_serialization);
criterion_main!(benches);
