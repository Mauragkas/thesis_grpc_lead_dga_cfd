//! SRP: gRPC server adapter. Translates proto messages into
//! `LocalRingMember` calls and the `MigrantBuffer` push. Holds no
//! business logic — pure wiring.

use crate::migration::buffer::MigrantBuffer;
use crate::proto::ring::ring_server::Ring as RingTrait;
use crate::proto::ring::{
    BoolMsg, Empty, FindSuccRequest, MigrateRequest, MigrateResponse, NodeInfo as ProtoNodeInfo,
    NodeInfoList, NotifyRequest, OptionalNodeInfo,
};
use crate::ring::member::LocalRingMember;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::debug;

pub struct RingServer {
    member: LocalRingMember,
    migrant_buffer: Arc<MigrantBuffer>,
}

impl RingServer {
    pub fn new(member: LocalRingMember, migrant_buffer: Arc<MigrantBuffer>) -> Self {
        Self {
            member,
            migrant_buffer,
        }
    }
}

fn to_proto(n: &crate::ring::state::NodeInfo) -> ProtoNodeInfo {
    ProtoNodeInfo {
        id: n.id,
        address: n.address.clone(),
    }
}

fn from_proto(n: ProtoNodeInfo) -> crate::ring::state::NodeInfo {
    crate::ring::state::NodeInfo {
        id: n.id,
        address: n.address,
    }
}

#[tonic::async_trait]
impl RingTrait for RingServer {
    async fn find_successor(
        &self,
        req: Request<FindSuccRequest>,
    ) -> Result<Response<ProtoNodeInfo>, Status> {
        let id = req.into_inner().id;
        let succ = self.member.find_successor(id).await;
        Ok(Response::new(to_proto(&succ)))
    }

    async fn get_predecessor(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<OptionalNodeInfo>, Status> {
        let pred = self.member.get_predecessor().await;
        Ok(Response::new(OptionalNodeInfo {
            node: pred.map(|p| to_proto(&p)),
        }))
    }

    async fn notify(&self, req: Request<NotifyRequest>) -> Result<Response<BoolMsg>, Status> {
        let other = req
            .into_inner()
            .other
            .ok_or_else(|| Status::invalid_argument("notify requires `other`"))?;
        let accepted = self.member.notify(from_proto(other)).await;
        Ok(Response::new(BoolMsg { ok: accepted }))
    }

    async fn get_successor_list(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<NodeInfoList>, Status> {
        let list = self.member.get_successor_list().await;
        Ok(Response::new(NodeInfoList {
            nodes: list.iter().map(to_proto).collect(),
        }))
    }

    async fn ping(&self, _req: Request<Empty>) -> Result<Response<BoolMsg>, Status> {
        Ok(Response::new(BoolMsg { ok: true }))
    }

    async fn migrate(
        &self,
        req: Request<MigrateRequest>,
    ) -> Result<Response<MigrateResponse>, Status> {
        let inner = req.into_inner();
        debug!(
            "Received {} migrants from {}",
            inner.individuals.len(),
            inner.sender
        );
        let migrants: Vec<_> = inner
            .individuals
            .into_iter()
            .map(|m| crate::migration::MigrantIndividual {
                genes: m.genes,
                fitness: m.fitness,
            })
            .collect();
        self.migrant_buffer.push(migrants).await;
        Ok(Response::new(MigrateResponse { accepted: true }))
    }
}
