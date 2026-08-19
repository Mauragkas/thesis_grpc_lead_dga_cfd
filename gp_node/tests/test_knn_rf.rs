use gp_node::{DatasetLoader, DatasetSplitter, KnnSurrogate, RfConfig, RfSurrogate};

#[test]
fn test_knn_and_rf_surrogates() {
    let candidates = [
        "tests/configs_and_scores.json",
        "../tests/configs_and_scores.json",
        "../../tests/configs_and_scores.json",
    ];
    let path = candidates.iter().find(|p| std::path::Path::new(p).exists()).unwrap();
    let records = DatasetLoader::load_from_json(path, Some(-15.0)).expect("Load dataset");
    let (x, y, n, dim) = DatasetLoader::extract_active_features(&records);

    let split = DatasetSplitter::split_60_20_20(&x, &y, n, dim, 42);

    // 1. Test k-NN Surrogate
    let knn = KnnSurrogate::fit(
        &split.train.x,
        &split.train.y,
        split.train.num_samples,
        dim,
        7,
    ).expect("k-NN fit failed");

    let knn_metrics = knn.evaluate(&split.test.x, &split.test.y).expect("k-NN eval");
    println!("k-NN Test Metrics: {:?}", knn_metrics);
    assert!(knn_metrics.r2_score > 0.40, "k-NN R^2 should be > 0.40, got {}", knn_metrics.r2_score);
    assert!(knn_metrics.rmse < 3.0, "k-NN RMSE should be < 3.0, got {}", knn_metrics.rmse);

    // 2. Test Random Forest Surrogate
    let rf_config = RfConfig::default();
    let rf = RfSurrogate::fit(
        &split.train.x,
        &split.train.y,
        split.train.num_samples,
        dim,
        rf_config,
    ).expect("RF fit failed");

    let rf_metrics = rf.evaluate(&split.test.x, &split.test.y).expect("RF eval");
    println!("RF Test Metrics: {:?}", rf_metrics);
    assert!(rf_metrics.r2_score > 0.55, "RF R^2 should be > 0.55, got {}", rf_metrics.r2_score);
    assert!(rf_metrics.rmse < 2.0, "RF RMSE should be < 2.0, got {}", rf_metrics.rmse);
}
