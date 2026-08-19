use surrogate_node::{
    BackendFactory, DatasetLoader, DatasetSplitter, GaussianProcessSurrogate, KernelType,
};
use std::path::Path;

#[test]
fn test_end_to_end_surrogate_training_and_evaluation() {
    let path = Path::new("../tests/configs_and_scores.json");
    if !path.exists() {
        eprintln!("Warning: Skipping end_to_end test because dataset was not found at {:?}", path);
        return;
    }

    // 1. Load dataset
    let records = DatasetLoader::load_from_json(path, Some(-15.0)).expect("Failed to load dataset");
    assert!(records.len() >= 100);

    // 2. Extract features
    let (x, y, n, dim) = DatasetLoader::extract_active_features(&records);

    // 3. Split 60-20-20
    let split = DatasetSplitter::split_60_20_20(&x, &y, n, dim, 42);
    assert_eq!(split.train.num_samples, (n * 6) / 10);
    assert_eq!(split.validation.num_samples, (n * 2) / 10);

    // 4. Create backend and fit model
    let backend = BackendFactory::create_best_backend();
    let mut surrogate = GaussianProcessSurrogate::new(backend, KernelType::Matern52);

    surrogate
        .fit(
            &split.train.x,
            &split.train.y,
            split.train.num_samples,
            split.train.dim,
            true,
        )
        .expect("Fit failed");

    // 5. Evaluate on holdout test partition
    let test_metrics = surrogate
        .evaluate(&split.test.x, &split.test.y)
        .expect("Evaluation failed");

    println!("Holdout Test R^2 Score: {:.4}", test_metrics.r2_score);
    println!("Holdout Test RMSE:      {:.4}", test_metrics.rmse);
    println!("Holdout Test MAE:       {:.4}", test_metrics.mae);

    // Assert that model learns well (R^2 > 0.70 and RMSE < 1.3)
    assert!(
        test_metrics.r2_score > 0.70,
        "Expected R^2 > 0.70 on holdout test set, got {:.4}",
        test_metrics.r2_score
    );
    assert!(
        test_metrics.rmse < 1.3,
        "Expected RMSE < 1.3, got {:.4}",
        test_metrics.rmse
    );
}
