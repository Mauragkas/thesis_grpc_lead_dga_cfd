use surrogate_node::{
    BackendFactory, DatasetLoader, DatasetSplitter, GaussianProcessSurrogate,
    KernelType, KnnSurrogate, MlpConfig, MlpSurrogate,
    RfConfig, RfSurrogate, ACTIVE_FEATURE_NAMES,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// JSON output schema
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelReport {
    name: String,
    device: String,
    /// Median training wall-clock time in milliseconds over N_TIMING_REPS fits.
    train_ms: f64,
    /// Median inference latency (ms) to predict 1,000 test points.
    infer_ms_per_1k: f64,
    /// Inferred throughput: evaluations per second.
    throughput_evals_per_sec: f64,
    val_r2: f64,
    val_rmse: f64,
    val_mae: f64,
    test_r2: f64,
    test_rmse: f64,
    test_mae: f64,
    test_max_residual: f64,
    /// Ground truth values on holdout test (for parity plot).
    y_test: Vec<f64>,
    /// Surrogate predictions on holdout test.
    y_pred: Vec<f64>,
    /// Learning curve: list of (n_train_used, test_r2) pairs.
    learning_curve: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComparisonReport {
    dataset_n_total: usize,
    n_train: usize,
    n_val: usize,
    n_test: usize,
    n_features: usize,
    feature_names: Vec<String>,
    models: Vec<ModelReport>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const N_TIMING_REPS: usize = 5;
const LEARNING_CURVE_SIZES: &[usize] = &[30, 60, 100, 150, 200, 250, 300];

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = samples.len() / 2;
    samples[mid]
}

fn resolve_dataset_path(args: &[String]) -> Option<String> {
    for arg in args {
        if !arg.starts_with("--") && Path::new(arg).exists() {
            return Some(arg.clone());
        }
    }
    for c in &["tests/configs_and_scores.json", "../tests/configs_and_scores.json", "../../tests/configs_and_scores.json"] {
        if Path::new(c).exists() { return Some(c.to_string()); }
    }
    None
}

fn resolve_output_path() -> String {
    // Emit next to this binary's workspace root
    for dir in &["scripts", "../scripts", "surrogate_node/scripts", "gp_node/scripts"] {
        if Path::new(dir).exists() {
            return format!("{}/compare_results.json", dir);
        }
    }
    "compare_results.json".to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark runners
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn bench_gp(
    x_train: &[f64], y_train: &[f64], n_train: usize, dim: usize,
    x_val: &[f64], y_val: &[f64],
    x_test: &[f64], y_test: &[f64],
    _x_full: &[f64], _y_full: &[f64],
) -> ModelReport {
    println!("  [GP] Training Gaussian Process (Matérn 5/2) ...");

    let mut train_times = Vec::with_capacity(N_TIMING_REPS);
    for _ in 0..N_TIMING_REPS {
        let backend = BackendFactory::create_best_backend();
        let mut gp = GaussianProcessSurrogate::new(backend, KernelType::Matern52);
        let t0 = Instant::now();
        gp.fit(x_train, y_train, n_train, dim, false).expect("GP fit failed");
        train_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // Final fitted model for evaluation
    let backend = BackendFactory::create_best_backend();
    let device_name = backend.device_name().to_string();
    let mut gp = GaussianProcessSurrogate::new(backend, KernelType::Matern52);
    gp.fit(x_train, y_train, n_train, dim, true).expect("GP fit failed");

    let val_m = gp.evaluate(x_val, y_val).expect("GP val eval failed");
    let test_m = gp.evaluate(x_test, y_test).expect("GP test eval failed");
    let (y_pred, _) = gp.predict(x_test).expect("GP predict failed");

    // Inference throughput: 1k predictions
    let x_test_1k: Vec<f64> = x_test.iter().cloned().cycle().take(1000 * dim).collect();
    let mut inf_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let t0 = Instant::now();
        let _ = gp.predict(&x_test_1k).expect("GP infer failed");
        inf_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let infer_ms = median_ms(inf_times);

    // Learning curve
    let mut learning_curve = Vec::new();
    for &sz in LEARNING_CURVE_SIZES.iter().filter(|&&s| s <= n_train) {
        let backend2 = BackendFactory::create_best_backend();
        let mut gp2 = GaussianProcessSurrogate::new(backend2, KernelType::Matern52);
        gp2.fit(&x_train[..sz*dim], &y_train[..sz], sz, dim, false).expect("GP lc fit");
        let m = gp2.evaluate(x_test, y_test).expect("GP lc eval");
        learning_curve.push((sz, m.r2_score));
    }

    ModelReport {
        name: "Gaussian Process (Matérn 5/2)".to_string(),
        device: device_name,
        train_ms: median_ms(train_times),
        infer_ms_per_1k: infer_ms,
        throughput_evals_per_sec: 1000.0 / (infer_ms / 1000.0),
        val_r2: val_m.r2_score, val_rmse: val_m.rmse, val_mae: val_m.mae,
        test_r2: test_m.r2_score, test_rmse: test_m.rmse, test_mae: test_m.mae,
        test_max_residual: test_m.max_residual,
        y_test: y_test.to_vec(), y_pred,
        learning_curve,
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_knn(
    x_train: &[f64], y_train: &[f64], n_train: usize, dim: usize,
    x_val: &[f64], y_val: &[f64],
    x_test: &[f64], y_test: &[f64],
    k: u32,
) -> ModelReport {
    println!("  [k-NN] Training k-NN (k={}) ...", k);

    let mut train_times = Vec::with_capacity(N_TIMING_REPS);
    let mut device_name = String::new();
    for _ in 0..N_TIMING_REPS {
        let t0 = Instant::now();
        let s = KnnSurrogate::fit(x_train, y_train, n_train, dim, k).expect("k-NN fit");
        train_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        if device_name.is_empty() { device_name = s.device_name().to_string(); }
    }

    let surrogate = KnnSurrogate::fit(x_train, y_train, n_train, dim, k).expect("k-NN fit");

    let val_m = surrogate.evaluate(x_val, y_val).expect("k-NN val eval");
    let test_m = surrogate.evaluate(x_test, y_test).expect("k-NN test eval");
    let (y_pred, _dist) = surrogate.predict(x_test).expect("k-NN predict");

    let x_test_1k: Vec<f64> = x_test.iter().cloned().cycle().take(1000 * dim).collect();
    let mut inf_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let t0 = Instant::now();
        let _ = surrogate.predict(&x_test_1k).expect("k-NN infer");
        inf_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let infer_ms = median_ms(inf_times);

    let mut learning_curve = Vec::new();
    for &sz in LEARNING_CURVE_SIZES.iter().filter(|&&s| s <= n_train) {
        let s2 = KnnSurrogate::fit(&x_train[..sz*dim], &y_train[..sz], sz, dim, k).expect("k-NN lc fit");
        let m = s2.evaluate(x_test, y_test).expect("k-NN lc eval");
        learning_curve.push((sz, m.r2_score));
    }

    ModelReport {
        name: format!("k-Nearest Neighbours (k={})", k),
        device: device_name,
        train_ms: median_ms(train_times),
        infer_ms_per_1k: infer_ms,
        throughput_evals_per_sec: 1000.0 / (infer_ms / 1000.0),
        val_r2: val_m.r2_score, val_rmse: val_m.rmse, val_mae: val_m.mae,
        test_r2: test_m.r2_score, test_rmse: test_m.rmse, test_mae: test_m.mae,
        test_max_residual: test_m.max_residual,
        y_test: y_test.to_vec(), y_pred,
        learning_curve,
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_rf(
    x_train: &[f64], y_train: &[f64], n_train: usize, dim: usize,
    x_val: &[f64], y_val: &[f64],
    x_test: &[f64], y_test: &[f64],
    config: RfConfig,
) -> ModelReport {
    println!("  [RF] Training Random Forest ({} trees, max_depth={}) ...", config.n_estimators, config.max_depth);

    let mut train_times = Vec::with_capacity(N_TIMING_REPS);
    let mut device_name = String::new();
    for _ in 0..N_TIMING_REPS {
        let t0 = Instant::now();
        let s = RfSurrogate::fit(x_train, y_train, n_train, dim, config).expect("RF fit");
        train_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        if device_name.is_empty() { device_name = s.device_name().to_string(); }
    }

    let surrogate = RfSurrogate::fit(x_train, y_train, n_train, dim, config).expect("RF fit");

    let val_m = surrogate.evaluate(x_val, y_val).expect("RF val eval");
    let test_m = surrogate.evaluate(x_test, y_test).expect("RF test eval");
    let y_pred = surrogate.predict(x_test).expect("RF predict");

    let x_test_1k: Vec<f64> = x_test.iter().cloned().cycle().take(1000 * dim).collect();
    let mut inf_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let t0 = Instant::now();
        let _ = surrogate.predict(&x_test_1k).expect("RF infer");
        inf_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let infer_ms = median_ms(inf_times);

    let mut learning_curve = Vec::new();
    for &sz in LEARNING_CURVE_SIZES.iter().filter(|&&s| s <= n_train) {
        let s2 = RfSurrogate::fit(&x_train[..sz*dim], &y_train[..sz], sz, dim, config).expect("RF lc fit");
        let m = s2.evaluate(x_test, y_test).expect("RF lc eval");
        learning_curve.push((sz, m.r2_score));
    }

    ModelReport {
        name: format!("Random Forest ({} trees, depth {})", config.n_estimators, config.max_depth),
        device: device_name,
        train_ms: median_ms(train_times),
        infer_ms_per_1k: infer_ms,
        throughput_evals_per_sec: 1000.0 / (infer_ms / 1000.0),
        val_r2: val_m.r2_score, val_rmse: val_m.rmse, val_mae: val_m.mae,
        test_r2: test_m.r2_score, test_rmse: test_m.rmse, test_mae: test_m.mae,
        test_max_residual: test_m.max_residual,
        y_test: y_test.to_vec(), y_pred,
        learning_curve,
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_mlp(
    x_train: &[f64], y_train: &[f64], n_train: usize, dim: usize,
    x_val: &[f64], y_val: &[f64],
    x_test: &[f64], y_test: &[f64],
    config: MlpConfig,
) -> ModelReport {
    println!("  [MLP] Training Neural Network ({} hidden layers: {:?}, epochs={}) ...",
        config.hidden_layers.len(), config.hidden_layers, config.epochs);

    let mut train_times = Vec::with_capacity(N_TIMING_REPS);
    let mut device_name = String::new();
    for _ in 0..N_TIMING_REPS {
        let t0 = Instant::now();
        let s = MlpSurrogate::fit(x_train, y_train, n_train, dim, config.clone()).expect("MLP fit");
        train_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        if device_name.is_empty() { device_name = s.device_name().to_string(); }
    }

    let surrogate = MlpSurrogate::fit(x_train, y_train, n_train, dim, config.clone()).expect("MLP fit");

    let val_m = surrogate.evaluate(x_val, y_val).expect("MLP val eval");
    let test_m = surrogate.evaluate(x_test, y_test).expect("MLP test eval");
    let y_pred = surrogate.predict(x_test).expect("MLP predict");

    let x_test_1k: Vec<f64> = x_test.iter().cloned().cycle().take(1000 * dim).collect();
    let mut inf_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let t0 = Instant::now();
        let _ = surrogate.predict(&x_test_1k).expect("MLP infer");
        inf_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let infer_ms = median_ms(inf_times);

    let mut learning_curve = Vec::new();
    for &sz in LEARNING_CURVE_SIZES.iter().filter(|&&s| s <= n_train) {
        let s2 = MlpSurrogate::fit(&x_train[..sz*dim], &y_train[..sz], sz, dim, config.clone()).expect("MLP lc fit");
        let m = s2.evaluate(x_test, y_test).expect("MLP lc eval");
        learning_curve.push((sz, m.r2_score));
    }

    ModelReport {
        name: format!("Neural Network MLP ({:?})", config.hidden_layers),
        device: device_name,
        train_ms: median_ms(train_times),
        infer_ms_per_1k: infer_ms,
        throughput_evals_per_sec: 1000.0 / (infer_ms / 1000.0),
        val_r2: val_m.r2_score, val_rmse: val_m.rmse, val_mae: val_m.mae,
        test_r2: test_m.r2_score, test_rmse: test_m.rmse, test_mae: test_m.mae,
        test_max_residual: test_m.max_residual,
        y_test: y_test.to_vec(), y_pred,
        learning_curve,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    let force_rerun = args.iter().any(|a| a == "--force" || a == "--rerun" || a == "-f");

    let data_path = match resolve_dataset_path(&args[1..]) {
        Some(p) => p,
        None => {
            eprintln!("Error: Dataset not found. Pass path as first argument or run from repo root.");
            std::process::exit(1);
        }
    };

    let out_path = resolve_output_path();

    // Load cached report if present
    let cached_report: Option<ComparisonReport> = if !force_rerun && Path::new(&out_path).exists() {
        fs::read_to_string(&out_path)
            .ok()
            .and_then(|content| serde_json::from_str::<ComparisonReport>(&content).ok())
    } else {
        None
    };

    println!("================================================================");
    println!("  SURROGATE MODEL COMPARISON: GP vs. k-NN vs. RF vs. MLP       ");
    println!("================================================================");

    println!("\n[1] Loading dataset '{}'...", data_path);
    let records = DatasetLoader::load_from_json(&data_path, Some(-15.0))
        .expect("Failed to load dataset");
    let (x_raw, y_raw, n_total, dim) = DatasetLoader::extract_active_features(&records);
    println!("    Loaded {} valid designs | {} active features", n_total, dim);

    println!("\n[2] Splitting 60/20/20 (seed 42)...");
    let split = DatasetSplitter::split_60_20_20(&x_raw, &y_raw, n_total, dim, 42);
    let n_train = split.train.num_samples;
    let n_val   = split.validation.num_samples;
    let n_test  = split.test.num_samples;
    println!("    Train: {} | Val: {} | Test: {}", n_train, n_val, n_test);

    println!("\n[3] Benchmarking surrogates ({} timing repetitions each)...", N_TIMING_REPS);

    let find_cached = |name_substr: &str| -> Option<ModelReport> {
        if let Some(ref cr) = cached_report {
            if let Some(m) = cr.models.iter().find(|m| m.name.to_lowercase().contains(&name_substr.to_lowercase())) {
                println!("  [Cached] Skipping {} (reused from '{}')", m.name, out_path);
                return Some(m.clone());
            }
        }
        None
    };

    let gp_report = find_cached("gaussian").unwrap_or_else(|| {
        bench_gp(
            &split.train.x, &split.train.y, n_train, dim,
            &split.validation.x, &split.validation.y,
            &split.test.x, &split.test.y,
            &x_raw, &y_raw,
        )
    });

    let knn_report = find_cached("k-nearest").unwrap_or_else(|| {
        bench_knn(
            &split.train.x, &split.train.y, n_train, dim,
            &split.validation.x, &split.validation.y,
            &split.test.x, &split.test.y,
            7,
        )
    });

    let rf_report = find_cached("random forest").unwrap_or_else(|| {
        bench_rf(
            &split.train.x, &split.train.y, n_train, dim,
            &split.validation.x, &split.validation.y,
            &split.test.x, &split.test.y,
            RfConfig::default(),
        )
    });

    let mlp_report = find_cached("neural network").or_else(|| find_cached("mlp")).unwrap_or_else(|| {
        bench_mlp(
            &split.train.x, &split.train.y, n_train, dim,
            &split.validation.x, &split.validation.y,
            &split.test.x, &split.test.y,
            MlpConfig::default(),
        )
    });

    let report = ComparisonReport {
        dataset_n_total: n_total,
        n_train, n_val, n_test,
        n_features: dim,
        feature_names: ACTIVE_FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
        models: vec![gp_report, knn_report, rf_report, mlp_report],
    };

    // Print summary table
    println!("\n{:=<80}", "");
    println!("{:^80}", "SURROGATE MODEL EVALUATION SUMMARY");
    println!("{:=<80}", "");
    println!("{:<35} | {:>8} | {:>8} | {:>8} | {:>10} | {:>12}",
        "Model", "Test R²", "RMSE", "MAE", "Train(ms)", "Infer 1k(ms)");
    println!("{:-<35}-+-{:-<8}-+-{:-<8}-+-{:-<8}-+-{:-<10}-+-{:-<12}", "", "", "", "", "", "");
    for m in &report.models {
        println!("{:<35} | {:>8.4} | {:>8.4} | {:>8.4} | {:>10.1} | {:>12.2}",
            m.name, m.test_r2, m.test_rmse, m.test_mae, m.train_ms, m.infer_ms_per_1k);
    }
    println!("{:=<80}", "");

    // Write JSON report
    let json = serde_json::to_string_pretty(&report).expect("Failed to serialise report");
    fs::write(&out_path, &json).expect("Failed to write JSON report");
    println!("\n[4] Wrote comparison report → '{}'", out_path);
    println!("    Run 'python3 scripts/compare_models.py' to generate plots.\n");
}
