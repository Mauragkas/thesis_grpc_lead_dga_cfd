mod api;
mod chord;
mod config;
mod ring;
mod storage;
mod transport;

use crate::storage::KeyStore;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

use crate::chord::ChordNode;
use crate::config::Config;
use crate::ring::{hash, NodeAddr};
use crate::storage::InMemoryStore;
use crate::transport::grpc::client::GrpcRemote;
use crate::transport::grpc::server;
use crate::transport::RemoteNode;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cfg = Config::from_env();
    let self_info = NodeAddr {
        id: hash(&cfg.self_uri),
        address: cfg.self_uri.clone(),
    };

    info!(
        "starting node id={} http={} grpc={} self_uri={} join={:?}",
        self_info.id, cfg.http_bind, cfg.grpc_bind, cfg.self_uri, cfg.join_uri
    );

    // Composition root: wire concrete implementations into abstractions (DIP).
    let storage = Arc::new(InMemoryStore::new());
    let remote = Arc::new(GrpcRemote::new());
    let chord = Arc::new(ChordNode::new(self_info.clone(), storage, remote));

    // Join an existing ring (with retries, since node1 may not be up yet).
    if let Some(ju) = cfg.join_uri.clone() {
        let c = chord.clone();
        tokio::spawn(async move {
            for _ in 0..60 {
                let s = c.successor().await;
                if s.id != c.self_info.id {
                    break;
                }
                c.join(&ju).await;
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }

    spawn_maintenance(chord.clone());

    // gRPC server (node-to-node communication).
    {
        let c = chord.clone();
        let addr = cfg.grpc_bind.clone();
        tokio::spawn(async move {
            info!("grpc listening on {}", addr);
            if let Err(e) = server::serve(c, &addr).await {
                tracing::error!("grpc server error: {e}");
            }
        });
    }

    // HTTP API (client-facing).
    let listener = tokio::net::TcpListener::bind(&cfg.http_bind).await.unwrap();
    info!("http listening on {}", cfg.http_bind);
    axum::serve(listener, api::router(chord)).await.unwrap();
}

fn spawn_maintenance<S, R>(c: Arc<ChordNode<S, R>>)
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    {
        let c = c.clone();
        tokio::spawn(async move {
            loop {
                c.stabilize().await;
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        });
    }
    {
        let c = c.clone();
        tokio::spawn(async move {
            loop {
                c.fix_fingers().await;
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        });
    }
    tokio::spawn(async move {
        loop {
            c.check_predecessor().await;
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });
}
