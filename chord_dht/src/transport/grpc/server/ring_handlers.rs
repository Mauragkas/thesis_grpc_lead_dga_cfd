use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::chord::ChordNode;
use crate::ring::NodeAddr;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::to_proto;
use crate::transport::grpc::gen::{
    BoolMsg, FindSuccRequest, GetPredRequest, NodeAddr as ProtoNode, NodeAddrList, NotifyRequest,
    OptionalNodeAddr, VidMsg,
};

pub(super) async fn find_successor<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<FindSuccRequest>,
) -> Result<Response<ProtoNode>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let inner = req.into_inner();
    let succ = node.find_successor(inner.vid, inner.id).await;
    Ok(Response::new(to_proto(succ)))
}

pub(super) async fn get_successor<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<VidMsg>,
) -> Result<Response<ProtoNode>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let vid = req.into_inner().vid;
    let s = node.successor(vid).await;
    Ok(Response::new(to_proto(s)))
}

pub(super) async fn get_successor_list<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<VidMsg>,
) -> Result<Response<NodeAddrList>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let vid = req.into_inner().vid;
    let list = node.successor_list(vid).await;
    let nodes = list.into_iter().map(to_proto).collect();
    Ok(Response::new(NodeAddrList { nodes }))
}

pub(super) async fn get_predecessor<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<GetPredRequest>,
) -> Result<Response<OptionalNodeAddr>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let vid = req.into_inner().vid;
    let pred = node.predecessor(vid).await.map(to_proto);
    Ok(Response::new(OptionalNodeAddr { node: pred }))
}

pub(super) async fn notify<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<NotifyRequest>,
) -> Result<Response<BoolMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let inner = req.into_inner();
    let other = match inner.other {
        Some(p) => NodeAddr {
            id: p.id,
            address: p.address,
        },
        None => return Err(Status::invalid_argument("missing other")),
    };
    node.notify(inner.vid, other).await;
    Ok(Response::new(BoolMsg { ok: true }))
}

pub(super) async fn ping() -> Result<Response<BoolMsg>, Status> {
    Ok(Response::new(BoolMsg { ok: true }))
}
