use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::chord::ChordNode;
use crate::ring::NodeAddr;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::gen::chord_server::{Chord, ChordServer};
use super::gen::{
    BoolMsg, Empty, KeyMsg, NodeAddr as ProtoNode, NodeAddrList, NodeIdMsg, OptionalNodeAddr,
    PutRequest, ValueMsg,
};

/// gRPC service that adapts incoming RPCs onto a `ChordNode`.
/// SRP: this layer only translates between wire types and the ring API;
/// the ring algorithm lives entirely in `ChordNode`.
pub struct ChordGrpcService<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    node: Arc<ChordNode<S, R>>,
}

impl<S, R> ChordGrpcService<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub fn new(node: Arc<ChordNode<S, R>>) -> Self {
        Self { node }
    }
}

fn to_proto(n: NodeAddr) -> ProtoNode {
    ProtoNode {
        id: n.id,
        address: n.address,
    }
}

#[tonic::async_trait]
impl<S, R> Chord for ChordGrpcService<S, R>
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    async fn find_successor(&self, req: Request<NodeIdMsg>) -> Result<Response<ProtoNode>, Status> {
        let id = req.into_inner().id;
        let succ = self.node.find_successor(id).await;
        Ok(Response::new(to_proto(succ)))
    }

    async fn get_successor(&self, _req: Request<Empty>) -> Result<Response<ProtoNode>, Status> {
        let s = self.node.successor().await;
        Ok(Response::new(to_proto(s)))
    }

    async fn get_successor_list(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<NodeAddrList>, Status> {
        let list = self.node.successor_list().await;
        let nodes = list.into_iter().map(to_proto).collect();
        Ok(Response::new(NodeAddrList { nodes }))
    }

    async fn get_predecessor(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<OptionalNodeAddr>, Status> {
        let pred = self.node.predecessor().await.map(to_proto);
        Ok(Response::new(OptionalNodeAddr { node: pred }))
    }

    async fn notify(&self, req: Request<ProtoNode>) -> Result<Response<BoolMsg>, Status> {
        let p = req.into_inner();
        let other = NodeAddr {
            id: p.id,
            address: p.address,
        };
        self.node.notify(other).await;
        Ok(Response::new(BoolMsg { ok: true }))
    }

    async fn get_local(&self, req: Request<KeyMsg>) -> Result<Response<ValueMsg>, Status> {
        let key = req.into_inner().key;
        match self.node.storage().get(&key).await {
            Some(v) => Ok(Response::new(ValueMsg { value: v })),
            None => Err(Status::not_found("key not found")),
        }
    }

    async fn put_local(&self, req: Request<PutRequest>) -> Result<Response<BoolMsg>, Status> {
        let r = req.into_inner();
        self.node.storage().put(r.key, r.value).await;
        Ok(Response::new(BoolMsg { ok: true }))
    }

    async fn delete_local(&self, req: Request<KeyMsg>) -> Result<Response<BoolMsg>, Status> {
        let key = req.into_inner().key;
        let ok = self.node.storage().remove(&key).await.is_some();
        Ok(Response::new(BoolMsg { ok }))
    }

    async fn ping(&self, _req: Request<Empty>) -> Result<Response<BoolMsg>, Status> {
        Ok(Response::new(BoolMsg { ok: true }))
    }
}

/// Spawn the gRPC server for a node.
pub async fn serve<S, R>(
    node: Arc<ChordNode<S, R>>,
    bind_addr: &str,
) -> Result<(), tonic::transport::Error>
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    let svc = ChordGrpcService::new(node);
    let addr: std::net::SocketAddr = bind_addr.parse().expect("invalid bind addr for grpc");
    tonic::transport::Server::builder()
        .add_service(ChordServer::new(svc))
        .serve(addr)
        .await
}
