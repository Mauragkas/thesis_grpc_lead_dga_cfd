use gp_node::{DatasetLoader, DatasetSplitter, StandardScaler, TargetScaler};
use std::collections::HashSet;
use std::path::Path;

#[test]
fn test_splitter_proportions_and_non_overlapping() {
    let num_samples = 100;
    let dim = 4;
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..num_samples {
        for d in 0..dim {
            x.push((i * 10 + d) as f64);
        }
        y.push(i as f64 * 2.5);
    }

    let split = DatasetSplitter::split_60_20_20(&x, &y, num_samples, dim, 12345);

    // 60-20-20 of 100 is 60, 20, 20
    assert_eq!(split.train.num_samples, 60);
    assert_eq!(split.validation.num_samples, 20);
    assert_eq!(split.test.num_samples, 20);

    assert_eq!(split.train.x.len(), 60 * dim);
    assert_eq!(split.validation.x.len(), 20 * dim);
    assert_eq!(split.test.x.len(), 20 * dim);

    // Verify all targets are unique and non-overlapping across partitions
    let train_targets: HashSet<i64> = split.train.y.iter().map(|&v| (v * 100.0) as i64).collect();
    let val_targets: HashSet<i64> = split.validation.y.iter().map(|&v| (v * 100.0) as i64).collect();
    let test_targets: HashSet<i64> = split.test.y.iter().map(|&v| (v * 100.0) as i64).collect();

    assert_eq!(train_targets.len(), 60);
    assert_eq!(val_targets.len(), 20);
    assert_eq!(test_targets.len(), 20);

    for target in &train_targets {
        assert!(!val_targets.contains(target));
        assert!(!test_targets.contains(target));
    }
    for target in &val_targets {
        assert!(!test_targets.contains(target));
    }
}

#[test]
fn test_standard_scaler_zero_mean_unit_variance() {
    let x_raw = vec![
        10.0, 100.0,
        20.0, 200.0,
        30.0, 300.0,
        40.0, 400.0,
        50.0, 500.0,
    ];
    let num_samples = 5;
    let dim = 2;

    let scaler = StandardScaler::fit(&x_raw, num_samples, dim);
    assert!((scaler.mean[0] - 30.0).abs() < 1e-10);
    assert!((scaler.mean[1] - 300.0).abs() < 1e-10);

    let x_scaled = scaler.transform(&x_raw);
    let num_elements = x_scaled.len() / dim;

    // Check zero mean
    let mean_col0: f64 = (0..num_elements).map(|i| x_scaled[i * dim]).sum::<f64>() / num_elements as f64;
    let mean_col1: f64 = (0..num_elements).map(|i| x_scaled[i * dim + 1]).sum::<f64>() / num_elements as f64;

    assert!(mean_col0.abs() < 1e-10);
    assert!(mean_col1.abs() < 1e-10);

    // Invert transform
    let x_recovered = scaler.inverse_transform(&x_scaled);
    for (orig, rec) in x_raw.iter().zip(x_recovered.iter()) {
        assert!((orig - rec).abs() < 1e-10);
    }
}

#[test]
fn test_target_scaler() {
    let y_raw = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let scaler = TargetScaler::fit(&y_raw);
    assert!((scaler.mean - 5.0).abs() < 1e-10);

    let y_scaled = scaler.transform(&y_raw);
    let y_recovered = scaler.inverse_transform(&y_scaled);

    for (orig, rec) in y_raw.iter().zip(y_recovered.iter()) {
        assert!((orig - rec).abs() < 1e-10);
    }
}

#[test]
fn test_dataset_loader_with_actual_file() {
    let path = Path::new("../tests/configs_and_scores.json");
    if path.exists() {
        let records = DatasetLoader::load_from_json(path, Some(-15.0)).unwrap();
        assert!(!records.is_empty());
        assert!(records.len() > 100);

        let (x, y, n, dim) = DatasetLoader::extract_active_features(&records);
        assert_eq!(dim, 11);
        assert_eq!(n, records.len());
        assert_eq!(x.len(), n * dim);
        assert_eq!(y.len(), n);
    }
}
