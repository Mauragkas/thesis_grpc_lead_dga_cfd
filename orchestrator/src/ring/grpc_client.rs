//! SRP: gRPC transport for `RingClient`. Caches channels per address
//! via `ChannelPool`. DIP: callers depend on `RingClient`, not this struct.

use crate::migration::MigrantIndividual;
use crate::proto::ring::ring_client::RingClient as GrpcRingClientProto;
use crate::proto::ring::{
    BoolMsg, Empty, FindSuccRequest, MigrateRequest, NodeInfo as ProtoNodeInfo, NodeInfoList,
    NotifyRequest, OptionalNodeInfo,
};
use crate::ring::client::RingClient;
use crate::ring::state::NodeInfo;
use crate::transport::ChannelPool;
use tonic::Status;
use tracing::warn;

pub struct GrpcRingClient {
    pool: ChannelPool,
}

impl GrpcRingClient {
    pub fn new() -> Self {
        Self {
            pool: ChannelPool::new(),
        }
    }
}

impl Default for GrpcRingClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl RingClient for GrpcRingClient {
    async fn find_successor(&self, addr: &str, id: u64) -> Result<NodeInfo, Status> {
        let ch = self.pool.get_or_connect(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let resp = c.find_successor(FindSuccRequest { id }).await?;
        Ok(NodeInfo::from(resp.into_inner()))
    }

    async fn get_predecessor(&self, addr: &str) -> Result<Option<NodeInfo>, Status> {
        let ch = self.pool.get_or_connect(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let resp: OptionalNodeInfo = c.get_predecessor(Empty {}).await?.into_inner();
        Ok(resp.node.map(NodeInfo::from))
    }

    async fn notify(&self, addr: &str, other: NodeInfo) -> Result<bool, Status> {
        let ch = self.pool.get_or_connect(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let resp: BoolMsg = c
            .notify(NotifyRequest {
                other: Some(ProtoNodeInfo::from(&other)),
            })
            .await?
            .into_inner();
        Ok(resp.ok)
    }

    async fn get_successor_list(&self, addr: &str) -> Result<Vec<NodeInfo>, Status> {
        let ch = self.pool.get_or_connect(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let resp: NodeInfoList = c.get_successor_list(Empty {}).await?.into_inner();
        Ok(resp.nodes.into_iter().map(NodeInfo::from).collect())
    }

    async fn ping(&self, addr: &str) -> Result<bool, Status> {
        match self.pool.get_or_connect(addr).await {
            Ok(ch) => {
                let mut c = GrpcRingClientProto::new(ch);
                let resp: BoolMsg = c.ping(Empty {}).await?.into_inner();
                Ok(resp.ok)
            }
            Err(e) => {
                warn!("ping failed for '{addr}': {e}");
                Ok(false)
            }
        }
    }

    async fn migrate(
        &self,
        addr: &str,
        sender: &str,
        migrants: &[MigrantIndividual],
    ) -> Result<bool, Status> {
        let ch = self.pool.get_or_connect(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let proto_migrants = migrants
            .iter()
            .map(crate::proto::ring::MigrantIndividual::from)
            .collect();
        let resp = c
            .migrate(MigrateRequest {
                individuals: proto_migrants,
                sender: sender.to_string(),
            })
            .await?;
        Ok(resp.into_inner().accepted)
    }
}
