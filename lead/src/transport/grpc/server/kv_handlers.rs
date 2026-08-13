use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::lead::LeadNode;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use crate::transport::grpc::gen::{
    BoolMsg, KeyList, KeyMsg, PutRequest, PutRoutedRequest, ValueMsg,
};

pub(super) async fn get_local<S, R>(
    node: &Arc<LeadNode<S, R>>,
    req: Request<KeyMsg>,
) -> Result<Response<ValueMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let key = req.into_inner().key;
    match node.storage().get(&key).await {
        Some(v) => Ok(Response::new(ValueMsg { value: v })),
        None => Err(Status::not_found("key not found")),
    }
}

pub(super) async fn put_local<S, R>(
    node: &Arc<LeadNode<S, R>>,
    req: Request<PutRequest>,
) -> Result<Response<BoolMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let r = req.into_inner();
    node.storage().put(r.key.clone(), r.value).await;
    node.record_insertion(&r.key).await;
    Ok(Response::new(BoolMsg { ok: true }))
}

pub(super) async fn delete_local<S, R>(
    node: &Arc<LeadNode<S, R>>,
    req: Request<KeyMsg>,
) -> Result<Response<BoolMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let key = req.into_inner().key;
    let ok = node.storage().remove(&key).await.is_some();
    Ok(Response::new(BoolMsg { ok }))
}

pub(super) async fn get_routed<S, R>(
    node: &Arc<LeadNode<S, R>>,
    req: Request<KeyMsg>,
) -> Result<Response<ValueMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let key = req.into_inner().key;
    let target = node.lookup_target(&key).await;
    let value = if target.address == node.self_uri {
        node.storage().get(&key).await
    } else {
        node.remote().get_local(&target.address, &key).await
    };
    match value {
        Some(v) => Ok(Response::new(ValueMsg { value: v })),
        None => Err(Status::not_found("key not found")),
    }
}

pub(super) async fn put_routed<S, R>(
    node: &Arc<LeadNode<S, R>>,
    req: Request<PutRoutedRequest>,
) -> Result<Response<BoolMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let r = req.into_inner();
    let target = node.lookup_target(&r.key).await;
    let ok = if target.address == node.self_uri {
        node.storage().put(r.key.clone(), r.value).await;
        node.record_insertion(&r.key).await;
        true
    } else {
        node.remote()
            .put_local(&target.address, &r.key, &r.value)
            .await
    };
    Ok(Response::new(BoolMsg { ok }))
}

pub(super) async fn get_keys<S, R>(node: &Arc<LeadNode<S, R>>) -> Result<Response<KeyList>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let keys: Vec<String> = node
        .storage()
        .snapshot()
        .await
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    Ok(Response::new(KeyList { keys }))
}
