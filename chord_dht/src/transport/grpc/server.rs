use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::chord::ChordNode;
use crate::ring::NodeAddr;
use crate::storage::KeyStore;
use crate::transport::{RangeResult, RemoteNode};

use super::gen::chord_server::{Chord, ChordServer};
use super::gen::KeyList;
use super::gen::{
    BoolMsg, Empty, FindSuccRequest, GetPredRequest, HeartbeatMsg, KeyMsg, ModelParams,
    ModelRequest, NodeAddr as ProtoNode, NodeAddrList, NotifyRequest, OptionalNodeAddr, PutRequest,
    PutRoutedRequest, RangeEntry, RangeForwardRequest, RangeRequest, RangeResponse, ValueMsg,
    VidMsg,
};

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

fn range_resp(r: RangeResult) -> RangeResponse {
    let entries: Vec<RangeEntry> = r
        .entries
        .into_iter()
        .map(|(k, v)| RangeEntry { key: k, value: v })
        .collect();
    RangeResponse {
        entries,
        complete: r.complete,
        next_address: r.next_address,
    }
}

#[tonic::async_trait]
impl<S, R> Chord for ChordGrpcService<S, R>
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    async fn get_keys(&self, _req: Request<Empty>) -> Result<Response<KeyList>, Status> {
        let keys: Vec<String> = self
            .node
            .storage()
            .snapshot()
            .await
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        Ok(Response::new(KeyList { keys }))
    }

    async fn put_routed(
        &self,
        req: Request<PutRoutedRequest>,
    ) -> Result<Response<BoolMsg>, Status> {
        let r = req.into_inner();
        let target = self.node.lookup_target(&r.key).await;
        let ok = if target.address == self.node.self_uri {
            self.node.storage().put(r.key.clone(), r.value).await;
            self.node.record_insertion(&r.key).await;
            true
        } else {
            self.node
                .remote()
                .put_local(&target.address, &r.key, &r.value)
                .await
        };
        Ok(Response::new(BoolMsg { ok }))
    }

    async fn get_routed(&self, req: Request<KeyMsg>) -> Result<Response<ValueMsg>, Status> {
        let key = req.into_inner().key;
        let target = self.node.lookup_target(&key).await;
        let value = if target.address == self.node.self_uri {
            self.node.storage().get(&key).await
        } else {
            self.node.remote().get_local(&target.address, &key).await
        };
        match value {
            Some(v) => Ok(Response::new(ValueMsg { value: v })),
            None => Err(Status::not_found("key not found")),
        }
    }

    async fn find_successor(
        &self,
        req: Request<FindSuccRequest>,
    ) -> Result<Response<ProtoNode>, Status> {
        let inner = req.into_inner();
        let succ = self.node.find_successor(inner.vid, inner.id).await;
        Ok(Response::new(to_proto(succ)))
    }

    async fn get_successor(&self, req: Request<VidMsg>) -> Result<Response<ProtoNode>, Status> {
        let vid = req.into_inner().vid;
        let s = self.node.successor(vid).await;
        Ok(Response::new(to_proto(s)))
    }

    async fn get_successor_list(
        &self,
        req: Request<VidMsg>,
    ) -> Result<Response<NodeAddrList>, Status> {
        let vid = req.into_inner().vid;
        let list = self.node.successor_list(vid).await;
        let nodes = list.into_iter().map(to_proto).collect();
        Ok(Response::new(NodeAddrList { nodes }))
    }

    async fn get_predecessor(
        &self,
        req: Request<GetPredRequest>,
    ) -> Result<Response<OptionalNodeAddr>, Status> {
        let vid = req.into_inner().vid;
        let pred = self.node.predecessor(vid).await.map(to_proto);
        Ok(Response::new(OptionalNodeAddr { node: pred }))
    }

    async fn notify(&self, req: Request<NotifyRequest>) -> Result<Response<BoolMsg>, Status> {
        let inner = req.into_inner();
        let other = match inner.other {
            Some(p) => NodeAddr {
                id: p.id,
                address: p.address,
            },
            None => return Err(Status::invalid_argument("missing other")),
        };
        self.node.notify(inner.vid, other).await;
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
        self.node.storage().put(r.key.clone(), r.value).await;
        self.node.record_insertion(&r.key).await;
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

    async fn range_query(
        &self,
        req: Request<RangeRequest>,
    ) -> Result<Response<RangeResponse>, Status> {
        let r = req.into_inner();
        let result = self
            .node
            .handle_range_query(&r.start_key, r.count, &r.caller_address)
            .await;
        Ok(Response::new(range_resp(result)))
    }

    async fn range_forward(
        &self,
        req: Request<RangeForwardRequest>,
    ) -> Result<Response<RangeResponse>, Status> {
        let r = req.into_inner();
        let payload: Vec<(String, String)> =
            r.payload.into_iter().map(|e| (e.key, e.value)).collect();
        let result = self
            .node
            .handle_range_forward(
                &r.start_key,
                r.count,
                &r.caller_address,
                r.origin_vid,
                payload,
            )
            .await;
        Ok(Response::new(range_resp(result)))
    }

    async fn push_model(&self, req: Request<ModelParams>) -> Result<Response<BoolMsg>, Status> {
        let r = req.into_inner();
        let ok = self.node.push_model(r.version, &r.data).await;
        Ok(Response::new(BoolMsg { ok }))
    }

    async fn request_model(
        &self,
        _req: Request<ModelRequest>,
    ) -> Result<Response<ModelParams>, Status> {
        let (version, data) = self.node.request_model().await;
        Ok(Response::new(ModelParams { version, data }))
    }

    async fn heartbeat(&self, req: Request<HeartbeatMsg>) -> Result<Response<BoolMsg>, Status> {
        let _h = req.into_inner();
        let ready = self.node.heartbeat().await;
        Ok(Response::new(BoolMsg { ok: ready }))
    }
}

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
