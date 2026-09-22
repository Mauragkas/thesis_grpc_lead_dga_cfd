use std::fs::OpenOptions;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use lead_node::api;
use lead_node::config::{Config, StorageBackend};
use lead_node::lead::LeadNode;
use lead_node::storage::{KeyStore, StorageEngine};
use lead_node::transport::grpc::{client::GrpcRemote, server};
use lead_node::transport::RemoteNode;

fn init_logging() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // 1. Stdout layer: human-readable, plain text for `docker logs`
    let stdout_layer = tracing_subscriber::fmt::layer().with_target(true);

    // 2. Optional JSON file layer: for Fluent-Bit
    let log_file_path = std::env::var("LOG_FILE_PATH").ok().or_else(|| {
        std::env::var("LOG_DIR").ok().map(|dir| {
            let name = std::env::var("CONTAINER_NAME")
                .or_else(|_| std::env::var("HOSTNAME"))
                .unwrap_or_else(|_| "lead_node".into());
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

#[tokio::main]
async fn main() {
    init_logging();

    let cfg = Config::from_env();
    info!(
        self_uri = %cfg.self_uri,
        http = %cfg.http_bind,
        grpc = %cfg.grpc_bind,
        vnodes = cfg.virtual_node_count,
        join = ?cfg.join_uri,
        storage_backend = ?cfg.storage_backend,
        storage_path = ?cfg.storage_path,
        "starting LEAD node"
    );

    let storage = match cfg.storage_backend {
        StorageBackend::Sled => {
            let path = cfg.storage_path.as_deref().unwrap_or("./data/lead");
            info!(path = %path, "initializing persistent sled storage");
            Arc::new(
                StorageEngine::sled(path)
                    .unwrap_or_else(|e| panic!("failed to open sled storage at {path}: {e}")),
            )
        }
        StorageBackend::Memory => {
            info!("initializing in-memory storage");
            Arc::new(StorageEngine::memory())
        }
    };
    let remote = Arc::new(GrpcRemote::new());
    let lead = Arc::new(LeadNode::new(cfg.clone(), storage, remote));
    lead.init_from_storage().await;

    let join_handle = if let Some(ju) = cfg.join_uri.clone() {
        let c = lead.clone();
        let retry_count = c.config.join_retry_count;
        let retry_delay = Duration::from_secs(c.config.join_retry_delay_secs);
        Some(tokio::spawn(async move {
            for _ in 0..retry_count {
                c.join(&ju).await;
                let mut pending = 0;
                for v in &c.vnodes {
                    if v.successor().await.id == v.vid {
                        pending += 1;
                    }
                }
                if pending == 0 {
                    info!("join complete: all vnodes have a real successor");
                    break;
                }
                info!("join: {pending} vnode(s) still alone, retrying");
                tokio::time::sleep(retry_delay).await;
            }
        }))
    } else {
        None
    };

    if let Some(handle) = join_handle {
        let _ = handle.await;
    }

    spawn_maintenance(lead.clone());

    {
        let c = lead.clone();
        let addr = cfg.grpc_bind.clone();
        tokio::spawn(async move {
            info!("grpc listening on {}", addr);
            if let Err(e) = server::serve(c, &addr).await {
                tracing::error!("grpc server error: {e}");
            }
        });
    }

    let listener = tokio::net::TcpListener::bind(&cfg.http_bind).await.unwrap();
    info!("http listening on {}", cfg.http_bind);
    axum::serve(listener, api::router(lead)).await.unwrap();
}

fn spawn_maintenance<S, R>(c: Arc<LeadNode<S, R>>)
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    {
        let c = c.clone();
        let interval = Duration::from_secs(c.config.stabilize_interval_secs);
        tokio::spawn(async move {
            loop {
                c.stabilize_all().await;
                tokio::time::sleep(interval).await;
            }
        });
    }
    {
        let c = c.clone();
        let interval = Duration::from_secs(c.config.fix_fingers_interval_secs);
        tokio::spawn(async move {
            loop {
                c.fix_fingers_all().await;
                tokio::time::sleep(interval).await;
            }
        });
    }
    {
        let c = c.clone();
        let interval = Duration::from_secs(c.config.check_predecessor_interval_secs);
        tokio::spawn(async move {
            loop {
                c.check_predecessor_all().await;
                tokio::time::sleep(interval).await;
            }
        });
    }
    {
        let c = c.clone();
        let interval = Duration::from_secs(c.config.heartbeat_interval_secs);
        tokio::spawn(async move {
            loop {
                c.heartbeat_round().await;
                tokio::time::sleep(interval).await;
            }
        });
    }
    {
        let c = c.clone();
        let interval = Duration::from_secs(c.config.maybe_retrain_interval_secs);
        tokio::spawn(async move {
            loop {
                c.maybe_retrain().await;
                tokio::time::sleep(interval).await;
            }
        });
    }
}
