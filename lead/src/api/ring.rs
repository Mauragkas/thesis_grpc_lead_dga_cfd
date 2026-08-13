use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};

use crate::ring::NodeAddr;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::AppState;

pub async fn get_successor<S, R>(State(node): State<AppState<S, R>>) -> Json<NodeAddr>
where
    S: KeyStore,
    R: RemoteNode,
{
    let vids = node.vids().await;
    Json(NodeAddr {
        id: vids[0],
        address: node.self_uri.clone(),
    })
}

pub async fn find_successor<S, R>(
    State(node): State<AppState<S, R>>,
    Path(id): Path<u64>,
) -> Json<NodeAddr>
where
    S: KeyStore,
    R: RemoteNode,
{
    let vids = node.vids().await;
    Json(node.find_successor(vids[0], id).await)
}

pub async fn get_predecessor<S, R>(State(node): State<AppState<S, R>>) -> Json<Option<NodeAddr>>
where
    S: KeyStore,
    R: RemoteNode,
{
    let vids = node.vids().await;
    Json(node.predecessor(vids[0]).await)
}

pub async fn notify<S, R>(
    State(node): State<AppState<S, R>>,
    Json(other): Json<NodeAddr>,
) -> StatusCode
where
    S: KeyStore,
    R: RemoteNode,
{
    let vids = node.vids().await;
    node.notify(vids[0], other).await;
    StatusCode::OK
}
