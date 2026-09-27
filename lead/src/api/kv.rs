use axum::extract::{Path, State};
use axum::http::StatusCode;

use crate::api::AppState;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

pub async fn get_local_kv<S, R>(
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

pub async fn put_local_kv<S, R>(
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

pub async fn del_local_kv<S, R>(
    State(node): State<AppState<S, R>>,
    Path(key): Path<String>,
) -> StatusCode
where
    S: KeyStore,
    R: RemoteNode,
{
    if node.storage().remove(&key).await.is_some() {
        node.learning.set_keys_total(node.storage().len().await);
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

pub async fn get_kv<S, R>(
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

pub async fn put_kv<S, R>(
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
    } else if node.remote().put_local(&target.address, &key, &body).await {
        StatusCode::CREATED
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

pub async fn del_kv<S, R>(State(node): State<AppState<S, R>>, Path(key): Path<String>) -> StatusCode
where
    S: KeyStore,
    R: RemoteNode,
{
    let target = node.lookup_target(&key).await;
    if target.address == node.self_uri {
        if node.storage().remove(&key).await.is_some() {
            node.learning.set_keys_total(node.storage().len().await);
            StatusCode::OK
        } else {
            StatusCode::NOT_FOUND
        }
    } else if node.remote().delete_local(&target.address, &key).await {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}
