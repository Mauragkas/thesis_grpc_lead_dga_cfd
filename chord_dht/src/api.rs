use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;

use crate::chord::ChordNode;
use crate::ring::NodeAddr;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

pub type AppState<S, R> = Arc<ChordNode<S, R>>;

pub fn router<S, R>(state: AppState<S, R>) -> Router
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    Router::new()
        .route("/successor", get(get_successor::<S, R>))
        .route("/health", get(|| async { "ok" }))
        .route("/find_successor/:id", get(find_successor::<S, R>))
        .route("/predecessor", get(get_predecessor::<S, R>))
        .route("/notify", post(notify::<S, R>))
        .route(
            "/kv/local/:key",
            get(get_local_kv::<S, R>)
                .post(put_local_kv::<S, R>)
                .delete(del_local_kv::<S, R>),
        )
        .route(
            "/kv/:key",
            get(get_kv::<S, R>)
                .post(put_kv::<S, R>)
                .delete(del_kv::<S, R>),
        )
        .route("/keys", get(get_all_keys::<S, R>))
        .route("/range", get(range_query::<S, R>))
        .with_state(state)
}

#[derive(Deserialize)]
struct RangeQ {
    key: String,
    count: u64,
}

async fn get_successor<S, R>(State(node): State<AppState<S, R>>) -> Json<NodeAddr>
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

async fn find_successor<S, R>(
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

async fn get_predecessor<S, R>(State(node): State<AppState<S, R>>) -> Json<Option<NodeAddr>>
where
    S: KeyStore,
    R: RemoteNode,
{
    let vids = node.vids().await;
    Json(node.predecessor(vids[0]).await)
}

async fn notify<S, R>(State(node): State<AppState<S, R>>, Json(other): Json<NodeAddr>) -> StatusCode
where
    S: KeyStore,
    R: RemoteNode,
{
    let vids = node.vids().await;
    node.notify(vids[0], other).await;
    StatusCode::OK
}

async fn get_local_kv<S, R>(
    State(node): State<AppState<S, R>>,
    Path(key): Path<String>,
) -> (StatusCode, String)
where
    S: KeyStore,
    R: RemoteNode,
{
    match node.storage().get(&key).await {
        Some(val) => (StatusCode::OK, val),
        None => (StatusCode::NOT_FOUND, "Key not found".into()),
    }
}

async fn put_local_kv<S, R>(
    State(node): State<AppState<S, R>>,
    Path(key): Path<String>,
    body: String,
) -> StatusCode
where
    S: KeyStore,
    R: RemoteNode,
{
    node.storage().put(key.clone(), body).await;
    node.record_insertion(&key).await;
    StatusCode::CREATED
}

async fn del_local_kv<S, R>(
    State(node): State<AppState<S, R>>,
    Path(key): Path<String>,
) -> StatusCode
where
    S: KeyStore,
    R: RemoteNode,
{
    if node.storage().remove(&key).await.is_some() {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn get_kv<S, R>(
    State(node): State<AppState<S, R>>,
    Path(key): Path<String>,
) -> (StatusCode, String)
where
    S: KeyStore,
    R: RemoteNode,
{
    let target = node.lookup_target(&key).await;
    if target.address == node.self_uri {
        match node.storage().get(&key).await {
            Some(v) => (StatusCode::OK, v),
            None => (StatusCode::NOT_FOUND, "Key not found".into()),
        }
    } else {
        match node.remote().get_local(&target.address, &key).await {
            Some(v) => (StatusCode::OK, v),
            None => (StatusCode::NOT_FOUND, "Key missing".into()),
        }
    }
}

async fn put_kv<S, R>(
    State(node): State<AppState<S, R>>,
    Path(key): Path<String>,
    body: String,
) -> StatusCode
where
    S: KeyStore,
    R: RemoteNode,
{
    let target = node.lookup_target(&key).await;
    if target.address == node.self_uri {
        node.storage().put(key.clone(), body).await;
        node.record_insertion(&key).await;
        StatusCode::CREATED
    } else {
        if node.remote().put_local(&target.address, &key, &body).await {
            StatusCode::CREATED
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

async fn del_kv<S, R>(State(node): State<AppState<S, R>>, Path(key): Path<String>) -> StatusCode
where
    S: KeyStore,
    R: RemoteNode,
{
    let target = node.lookup_target(&key).await;
    if target.address == node.self_uri {
        if node.storage().remove(&key).await.is_some() {
            StatusCode::OK
        } else {
            StatusCode::NOT_FOUND
        }
    } else {
        if node.remote().delete_local(&target.address, &key).await {
            StatusCode::OK
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

async fn get_all_keys<S, R>(State(node): State<AppState<S, R>>) -> Json<Vec<(String, String)>>
where
    S: KeyStore,
    R: RemoteNode,
{
    Json(node.storage().snapshot().await)
}

async fn range_query<S, R>(
    State(node): State<AppState<S, R>>,
    Query(q): Query<RangeQ>,
) -> Json<Vec<(String, String)>>
where
    S: KeyStore,
    R: RemoteNode,
{
    let result = node.range_query(&q.key, q.count, &node.self_uri).await;
    Json(result.entries)
}
