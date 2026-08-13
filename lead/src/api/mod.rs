use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;

use crate::lead::LeadNode;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

mod kv;
mod range;
mod ring;

pub type AppState<S, R> = Arc<LeadNode<S, R>>;

pub fn router<S, R>(state: AppState<S, R>) -> Router
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    Router::new()
        .route("/successor", get(ring::get_successor::<S, R>))
        .route("/health", get(|| async { "ok" }))
        .route("/find_successor/:id", get(ring::find_successor::<S, R>))
        .route("/predecessor", get(ring::get_predecessor::<S, R>))
        .route("/notify", post(ring::notify::<S, R>))
        .route(
            "/kv/local/:key",
            get(kv::get_local_kv::<S, R>)
                .post(kv::put_local_kv::<S, R>)
                .delete(kv::del_local_kv::<S, R>),
        )
        .route(
            "/kv/:key",
            get(kv::get_kv::<S, R>)
                .post(kv::put_kv::<S, R>)
                .delete(kv::del_kv::<S, R>),
        )
        .route("/keys", get(range::get_all_keys::<S, R>))
        .route("/range", get(range::range_query::<S, R>))
        .with_state(state)
}
