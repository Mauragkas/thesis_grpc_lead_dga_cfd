use axum::{
    extract::{Query, State},
    Json,
};
use serde::Deserialize;

use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::AppState;

#[derive(Deserialize)]
pub struct RangeQ {
    pub key: String,
    pub count: u64,
}

pub async fn get_all_keys<S, R>(State(node): State<AppState<S, R>>) -> Json<Vec<(String, String)>>
where
    S: KeyStore,
    R: RemoteNode,
{
    Json(node.storage().snapshot().await)
}

pub async fn range_query<S, R>(
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
