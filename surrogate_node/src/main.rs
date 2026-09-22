use std::env;
use std::fs::OpenOptions;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use surrogate_node::proto::surrogate::surrogate_service_server::SurrogateServiceServer;
use surrogate_node::{
    BackendFactory, DatasetLoader, DatasetSplitter, GaussianProcessSurrogate, KernelType,
    SurrogateConfig, SurrogateServer, SurrogateState, ACTIVE_FEATURE_NAMES,
};
use tonic::transport::Server;
use tracing::{error, info};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

fn init_logging() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // 1. Stdout layer: human-readable, plain text for `docker logs`
    let stdout_layer = tracing_subscriber::fmt::layer().with_target(true);

    // 2. Optional JSON file layer: for Fluent-Bit
    let log_file_path = std::env::var("LOG_FILE_PATH").ok().or_else(|| {
        std::env::var("LOG_DIR").ok().map(|dir| {
            let name = std::env::var("CONTAINER_NAME")
                .or_else(|_| std::env::var("HOSTNAME"))
                .unwrap_or_else(|_| "surrogate_node".into());
            format!("{dir}/{name}.log")
        })
    });

    if let Some(path_str) = log_file_path {
        let path = Path::new(&path_str);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(file) = OpenOptions::new().create(true).append(true).open(path) {
            let file_layer = tracing_subscriber::fmt::layer()
                .json()
                .with_target(true)
                .with_writer(Mutex::new(file));

            tracing_subscriber::registry()
                .with(env_filter)
                .with(stdout_layer)
                .with(file_layer)
                .init();
            return;
        }
    }

    tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer)
        .init();
}

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

fn run_cli_benchmark(data_path: &str) {
    println!("============================================================");
    println!("   GAUSSIAN PROCESS SURROGATE NODE (CUDA/ROCm/OpenMP)       ");
    println!("============================================================");

    println!("\n[1] Probing Compute Hardware Devices...");
    let devices = BackendFactory::probe_devices();
    for (i, dev) in devices.iter().enumerate() {
        println!("    Device #{}: {}", i, dev);
    }

    let backend = BackendFactory::create_best_backend();
    println!("    Selected Compute Engine: {}", backend.device_name());

    println!("\n[2] Loading Dataset from '{}'...", data_path);
    let records = match DatasetLoader::load_from_json(data_path, Some(-15.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error loading dataset: {}", e);
            std::process::exit(1);
        }
    };
    println!(
        "    Total valid converged configurations: {}",
        records.len()
    );

    let (x_raw, y_raw, num_samples, dim) = DatasetLoader::extract_active_features(&records);
    println!(
        "    Extracted {} features per sample: {:?}",
        dim, ACTIVE_FEATURE_NAMES
    );

    println!("\n[3] Splitting Dataset (60% Train, 20% Val, 20% Test)...");
    let split = DatasetSplitter::split_60_20_20(&x_raw, &y_raw, num_samples, dim, 42);
    println!("    Train set:      {} samples", split.train.num_samples);
    println!(
        "    Validation set: {} samples",
        split.validation.num_samples
    );
    println!("    Test set:       {} samples", split.test.num_samples);

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

    let val_metrics = surrogate
        .evaluate(&split.validation.x, &split.validation.y)
        .expect("Validation evaluation failed");

    let test_metrics = surrogate
        .evaluate(&split.test.x, &split.test.y)
        .expect("Test evaluation failed");

    println!("------------------------------------------------------------");
    println!("                 MODEL EVALUATION SUMMARY                   ");
    println!("------------------------------------------------------------");
    println!(
        "{:<20} | {:<12} | {:<12}",
        "Metric", "Validation", "Test (Holdout)"
    );
    println!("---------------------+--------------+-------------");
    println!(
        "{:<20} | {:<12.4} | {:<12.4}",
        "R^2 Score", val_metrics.r2_score, test_metrics.r2_score
    );
    println!(
        "{:<20} | {:<12.4} | {:<12.4}",
        "RMSE", val_metrics.rmse, test_metrics.rmse
    );
    println!(
        "{:<20} | {:<12.4} | {:<12.4}",
        "MAE", val_metrics.mae, test_metrics.mae
    );
    println!(
        "{:<20} | {:<12.4} | {:<12.4}",
        "Max Residual", val_metrics.max_residual, test_metrics.max_residual
    );
    println!("------------------------------------------------------------\n");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && (args[1] == "--bench" || args[1].ends_with(".json")) {
        let path = if args[1] == "--bench" {
            args.get(2).map(|s| s.as_str())
        } else {
            Some(args[1].as_str())
        };
        if let Some(p) = resolve_dataset_path(path) {
            run_cli_benchmark(&p);
            return Ok(());
        }
    }

    init_logging();
    info!("Starting Surrogate Node gRPC Microservice...");

    let config = SurrogateConfig::from_env();
    let addr: SocketAddr = config.grpc_bind.parse().map_err(|e| {
        error!("Invalid GRPC_BIND '{}': {e}", config.grpc_bind);
        e
    })?;

    let state = SurrogateState::new(config);
    let server = SurrogateServer::new(state);

    info!("Surrogate gRPC service listening on {}", addr);
    Server::builder()
        .add_service(SurrogateServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
