mod kv_handlers;
mod model_handlers;
mod range_handlers;
mod ring_handlers;

use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::lead::LeadNode;
use crate::ring::NodeAddr;
use crate::storage::KeyStore;
use crate::transport::{RangeResult, RemoteNode};

use super::gen::lead_server::{Lead, LeadServer};
use super::gen::{
    BoolMsg, DeliverRangeRequest, Empty, FindSuccRequest, GetPredRequest, HeartbeatMsg, KeyList,
    KeyMsg, ModelDiff, ModelDiffRequest, ModelParams, ModelRequest, NodeAddr as ProtoNode,
    NodeAddrList, NotifyRequest, OptionalNodeAddr, PruneRequest, PutRequest, PutRoutedRequest,
    RangeEntry, RangeForwardRequest, RangeRequest, RangeResponse, ResourceReport, ValueMsg, VidMsg,
};

pub struct LeadGrpcService<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    node: Arc<LeadNode<S, R>>,
}

impl<S, R> LeadGrpcService<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub fn new(node: Arc<LeadNode<S, R>>) -> Self {
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
impl<S, R> Lead for LeadGrpcService<S, R>
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    async fn find_successor(
        &self,
        req: Request<FindSuccRequest>,
    ) -> Result<Response<ProtoNode>, Status> {
        ring_handlers::find_successor(&self.node, req).await
    }
    async fn get_successor(&self, req: Request<VidMsg>) -> Result<Response<ProtoNode>, Status> {
        ring_handlers::get_successor(&self.node, req).await
    }
    async fn get_successor_list(
        &self,
        req: Request<VidMsg>,
    ) -> Result<Response<NodeAddrList>, Status> {
        ring_handlers::get_successor_list(&self.node, req).await
    }
    async fn get_predecessor(
        &self,
        req: Request<GetPredRequest>,
    ) -> Result<Response<OptionalNodeAddr>, Status> {
        ring_handlers::get_predecessor(&self.node, req).await
    }
    async fn notify(&self, req: Request<NotifyRequest>) -> Result<Response<BoolMsg>, Status> {
        ring_handlers::notify(&self.node, req).await
    }
    async fn ping(&self, _req: Request<Empty>) -> Result<Response<BoolMsg>, Status> {
        ring_handlers::ping().await
    }

    async fn get_local(&self, req: Request<KeyMsg>) -> Result<Response<ValueMsg>, Status> {
        kv_handlers::get_local(&self.node, req).await
    }
    async fn put_local(&self, req: Request<PutRequest>) -> Result<Response<BoolMsg>, Status> {
        kv_handlers::put_local(&self.node, req).await
    }
    async fn delete_local(&self, req: Request<KeyMsg>) -> Result<Response<BoolMsg>, Status> {
        kv_handlers::delete_local(&self.node, req).await
    }
    async fn get_routed(&self, req: Request<KeyMsg>) -> Result<Response<ValueMsg>, Status> {
        kv_handlers::get_routed(&self.node, req).await
    }
    async fn put_routed(
        &self,
        req: Request<PutRoutedRequest>,
    ) -> Result<Response<BoolMsg>, Status> {
        kv_handlers::put_routed(&self.node, req).await
    }
    async fn get_keys(&self, _req: Request<Empty>) -> Result<Response<KeyList>, Status> {
        kv_handlers::get_keys(&self.node).await
    }

    async fn range_query(
        &self,
        req: Request<RangeRequest>,
    ) -> Result<Response<RangeResponse>, Status> {
        range_handlers::range_query(&self.node, req).await
    }
    async fn range_forward(
        &self,
        req: Request<RangeForwardRequest>,
    ) -> Result<Response<RangeResponse>, Status> {
        range_handlers::range_forward(&self.node, req).await
    }
    async fn deliver_range(
        &self,
        req: Request<DeliverRangeRequest>,
    ) -> Result<Response<BoolMsg>, Status> {
        range_handlers::deliver_range(req).await
    }

    async fn push_model(&self, req: Request<ModelParams>) -> Result<Response<BoolMsg>, Status> {
        model_handlers::push_model(&self.node, req).await
    }
    async fn request_model(
        &self,
        _req: Request<ModelRequest>,
    ) -> Result<Response<ModelParams>, Status> {
        model_handlers::request_model(&self.node).await
    }
    async fn get_model_diff(
        &self,
        req: Request<ModelDiffRequest>,
    ) -> Result<Response<ModelDiff>, Status> {
        model_handlers::get_model_diff(&self.node, req).await
    }
    async fn heartbeat(&self, req: Request<HeartbeatMsg>) -> Result<Response<BoolMsg>, Status> {
        model_handlers::heartbeat(&self.node, req).await
    }
    async fn report_resources(
        &self,
        req: Request<ResourceReport>,
    ) -> Result<Response<BoolMsg>, Status> {
        model_handlers::report_resources(req).await
    }
    async fn prune_vnode(&self, req: Request<PruneRequest>) -> Result<Response<BoolMsg>, Status> {
        model_handlers::prune_vnode(&self.node, req).await
    }
}

pub async fn serve<S, R>(
    node: Arc<LeadNode<S, R>>,
    bind_addr: &str,
) -> Result<(), tonic::transport::Error>
where
    S: KeyStore + 'static,
    R: RemoteNode + 'static,
{
    let svc = LeadGrpcService::new(node);
    let addr: std::net::SocketAddr = bind_addr.parse().expect("invalid bind addr for grpc");
    tonic::transport::Server::builder()
        .add_service(LeadServer::new(svc))
        .serve(addr)
        .await
}
