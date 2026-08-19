use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use surrogate_node::{
    ComputeBackend, CpuOpenMpBackend, CudaBackend, GaussianProcessSurrogate, GpHyperparameters,
    KernelType,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

fn generate_synthetic_data(num_samples: usize, dim: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(num_samples * dim);
    let mut y = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut row_sum = 0.0;
        for _ in 0..dim {
            let val = rng.gen_range(-2.0..2.0);
            x.push(val);
            row_sum += val;
        }
        y.push(row_sum * 0.5 + rng.gen_range(-0.1..0.1));
    }

    (x, y)
}

fn bench_covariance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("covariance_matrix");
    let dim = 11;
    let params = GpHyperparameters::new(vec![1.0; dim], 1.5, 1e-3, KernelType::Matern52);
    let cpu_backend = CpuOpenMpBackend::new().unwrap();
    let cuda_backend = CudaBackend::new(0).ok();

    for &n in &[100, 360, 600, 1000] {
        let (x, _) = generate_synthetic_data(n, dim, 42);

        group.bench_with_input(BenchmarkId::new("CPU_OpenMP", n), &n, |b, &size| {
            b.iter(|| {
                cpu_backend
                    .compute_covariance(
                        black_box(&x),
                        black_box(size),
                        black_box(&x),
                        black_box(size),
                        black_box(dim),
                        black_box(&params),
                        black_box(true),
                    )
                    .unwrap()
            });
        });

        if let Some(ref cuda) = cuda_backend {
            group.bench_with_input(BenchmarkId::new("CUDA_GPU", n), &n, |b, &size| {
                b.iter(|| {
                    cuda.compute_covariance(
                        black_box(&x),
                        black_box(size),
                        black_box(&x),
                        black_box(size),
                        black_box(dim),
                        black_box(&params),
                        black_box(true),
                    )
                    .unwrap()
                });
            });
        }
    }
    group.finish();
}

fn bench_cholesky_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_decomposition");
    let dim = 11;
    let params = GpHyperparameters::new(vec![1.0; dim], 1.5, 1e-3, KernelType::Matern52);
    let cpu_backend = CpuOpenMpBackend::new().unwrap();

    for &n in &[100, 360, 600, 1000] {
        let (x, _) = generate_synthetic_data(n, dim, 42);
        let k = cpu_backend
            .compute_covariance(&x, n, &x, n, dim, &params, true)
            .unwrap();

        group.bench_with_input(BenchmarkId::new("Cholesky_L", n), &n, |b, &size| {
            b.iter(|| {
                cpu_backend
                    .cholesky(black_box(&k), black_box(size))
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_predict_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_batch");
    let dim = 11;
    let n_train = 360;
    let (x_train, y_train) = generate_synthetic_data(n_train, dim, 42);

    let cpu_backend = Box::new(CpuOpenMpBackend::new().unwrap());
    let mut surrogate = GaussianProcessSurrogate::new(cpu_backend, KernelType::Matern52);
    surrogate
        .fit(&x_train, &y_train, n_train, dim, false)
        .unwrap();

    for &m in &[100, 500, 1000, 2000] {
        let (x_test, _) = generate_synthetic_data(m, dim, 999);

        group.bench_with_input(BenchmarkId::new("Inference_M_samples", m), &m, |b, _| {
            b.iter(|| surrogate.predict(black_box(&x_test)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_covariance_matrix,
    bench_cholesky_decomposition,
    bench_predict_batch
);
criterion_main!(benches);
