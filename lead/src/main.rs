use std::sync::Arc;
use std::time::Duration;

use tracing::info;
use tracing_subscriber::EnvFilter;

use lead_node::api;
use lead_node::config::Config;
use lead_node::lead::LeadNode;
use lead_node::storage::{InMemoryStore, KeyStore};
use lead_node::transport::grpc::{client::GrpcRemote, server};
use lead_node::transport::RemoteNode;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .with_target(true)
        .init();

    let cfg = Config::from_env();
    info!(
        self_uri = %cfg.self_uri,
        http = %cfg.http_bind,
        grpc = %cfg.grpc_bind,
        vnodes = cfg.virtual_node_count,
        join = ?cfg.join_uri,
        "starting LEAD node"
    );

    let storage = Arc::new(InMemoryStore::new());
    let remote = Arc::new(GrpcRemote::new());
    let lead = Arc::new(LeadNode::new(
        cfg.self_uri.clone(),
        cfg.virtual_node_count.max(1),
        storage,
        remote,
    ));

    let join_handle = if let Some(ju) = cfg.join_uri.clone() {
        let c = lead.clone();
        Some(tokio::spawn(async move {
            for _ in 0..300 {
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
                tokio::time::sleep(Duration::from_secs(1)).await;
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
        tokio::spawn(async move {
            loop {
                c.stabilize_all().await;
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });
    }
    {
        let c = c.clone();
        tokio::spawn(async move {
            loop {
                c.fix_fingers_all().await;
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    }
    {
        let c = c.clone();
        tokio::spawn(async move {
            loop {
                c.check_predecessor_all().await;
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        });
    }
    {
        let c = c.clone();
        tokio::spawn(async move {
            loop {
                c.heartbeat_round().await;
                tokio::time::sleep(Duration::from_secs(8)).await;
            }
        });
    }
    {
        let c = c.clone();
        tokio::spawn(async move {
            loop {
                c.maybe_retrain().await;
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    }
    // {
    //     let c = c.clone();
    //     tokio::spawn(async move {
    //         loop {
    //             c.prune_low_throughput_vnodes().await;
    //             tokio::time::sleep(Duration::from_secs(30)).await;
    //         }
    //     });
    // }
}
