use surrogate_node::{
    BackendFactory, DatasetLoader, DatasetSplitter, GaussianProcessSurrogate,
    KernelType, ACTIVE_FEATURE_NAMES,
};
use std::env;
use std::path::Path;
use std::time::Instant;

fn resolve_dataset_path(arg_path: Option<&str>) -> Option<String> {
    if let Some(p) = arg_path {
        if Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    let candidates = [
        "tests/configs_and_scores.json",
        "../tests/configs_and_scores.json",
        "../../tests/configs_and_scores.json",
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return Some(c.to_string());
        }
    }
    None
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let data_path = match resolve_dataset_path(args.get(1).map(|s| s.as_str())) {
        Some(p) => p,
        None => {
            eprintln!("Error: Dataset file configs_and_scores.json not found in candidate paths.");
            std::process::exit(1);
        }
    };

    println!("============================================================");
    println!("   GAUSSIAN PROCESS SURROGATE NODE (CUDA/ROCm/OpenMP)       ");
    println!("============================================================");

    // 1. Hardware capability detection
    println!("\n[1] Probing Compute Hardware Devices...");
    let devices = BackendFactory::probe_devices();
    for (i, dev) in devices.iter().enumerate() {
        println!("    Device #{}: {}", i, dev);
    }

    let backend = BackendFactory::create_best_backend();
    println!("    Selected Compute Engine: {}", backend.device_name());

    // 2. Load and validate dataset
    println!("\n[2] Loading Dataset from '{}'...", data_path);
    let records = match DatasetLoader::load_from_json(&data_path, Some(-15.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error loading dataset: {}", e);
            std::process::exit(1);
        }
    };
    println!("    Total valid converged configurations: {}", records.len());

    // 3. Extract 11 active aerodynamic features
    let (x_raw, y_raw, num_samples, dim) = DatasetLoader::extract_active_features(&records);
    println!("    Extracted {} features per sample: {:?}", dim, ACTIVE_FEATURE_NAMES);

    // 4. Split dataset into 60% Train, 20% Val, 20% Test
    println!("\n[3] Splitting Dataset (60% Train, 20% Val, 20% Test)...");
    let split = DatasetSplitter::split_60_20_20(&x_raw, &y_raw, num_samples, dim, 42);
    println!("    Train set:      {} samples", split.train.num_samples);
    println!("    Validation set: {} samples", split.validation.num_samples);
    println!("    Test set:       {} samples", split.test.num_samples);

    // 5. Fit Gaussian Process Surrogate Model
    println!("\n[4] Training Gaussian Process Surrogate Model (Matérn 5/2)...");
    let start_fit = Instant::now();
    let mut surrogate = GaussianProcessSurrogate::new(backend, KernelType::Matern52);

    surrogate
        .fit(
            &split.train.x,
            &split.train.y,
            split.train.num_samples,
            split.train.dim,
            true,
        )
        .expect("GP fitting failed");

    let fit_duration = start_fit.elapsed();
    println!("    Model fitted successfully in {:.2?}", fit_duration);
    println!("    Optimized Hyperparameters:");
    println!("      Signal Variance: {:.4}", surrogate.hyperparameters().signal_variance);
    println!("      Noise Variance:  {:.6}", surrogate.hyperparameters().noise_variance);

    // 6. Validation and Holdout Test Evaluation
    println!("\n[5] Evaluating Holdout Test Performance...");
    let val_metrics = surrogate
        .evaluate(&split.validation.x, &split.validation.y)
        .expect("Validation evaluation failed");

    let test_metrics = surrogate
        .evaluate(&split.test.x, &split.test.y)
        .expect("Test evaluation failed");

    println!("------------------------------------------------------------");
    println!("                 MODEL EVALUATION SUMMARY                   ");
    println!("------------------------------------------------------------");
    println!("{:<20} | {:<12} | {:<12}", "Metric", "Validation", "Test (Holdout)");
    println!("---------------------+--------------+-------------");
    println!("{:<20} | {:<12.4} | {:<12.4}", "R^2 Score", val_metrics.r2_score, test_metrics.r2_score);
    println!("{:<20} | {:<12.4} | {:<12.4}", "RMSE", val_metrics.rmse, test_metrics.rmse);
    println!("{:<20} | {:<12.4} | {:<12.4}", "MAE", val_metrics.mae, test_metrics.mae);
    println!("{:<20} | {:<12.4} | {:<12.4}", "Max Residual", val_metrics.max_residual, test_metrics.max_residual);
    println!("------------------------------------------------------------");

    // 7. Sample Predictions
    println!("\n[6] Sample Test Set Predictions (First 5 Holdout Designs):");
    let (sample_means, sample_stds) = surrogate
        .predict(&split.test.x[0..5 * dim])
        .expect("Sample predictions failed");

    println!("{:<6} | {:<14} | {:<14} | {:<14} | {:<10}", "#", "Ground Truth", "Predicted Mean", "Uncertainty (σ)", "Error");
    println!("-------+----------------+----------------+----------------+-----------");
    for i in 0..5 {
        let yt = split.test.y[i];
        let yp = sample_means[i];
        let unc = sample_stds[i];
        let err = (yt - yp).abs();
        println!("{:<6} | {:<14.4} | {:<14.4} | {:<14.4} | {:<10.4}", i + 1, yt, yp, unc, err);
    }
    println!("============================================================\n");
}
