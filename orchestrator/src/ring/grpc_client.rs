//! SRP: gRPC transport for `RingClient`. Caches channels per address
//! for cheap reuse. DIP: callers depend on `RingClient`, not this struct.

use crate::proto::ring::ring_client::RingClient as GrpcRingClientProto;
use crate::proto::ring::{
    BoolMsg, Empty, FindSuccRequest, NodeInfo as ProtoNodeInfo, NodeInfoList, NotifyRequest,
    OptionalNodeInfo,
};
use crate::ring::client::RingClient;
use crate::ring::state::NodeInfo;
use std::collections::HashMap;
use tonic::transport::Channel;
use tonic::Status;
use tracing::warn;

pub struct GrpcRingClient {
    channels: tokio::sync::Mutex<HashMap<String, Channel>>,
}

impl GrpcRingClient {
    pub fn new() -> Self {
        Self {
            channels: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    async fn channel(&self, addr: &str) -> Result<Channel, Status> {
        let mut cache = self.channels.lock().await;
        if let Some(ch) = cache.get(addr) {
            return Ok(ch.clone());
        }
        let uri = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr.to_string()
        } else {
            format!("http://{addr}")
        };
        let ch = Channel::from_shared(uri)
            .map_err(|e| Status::invalid_argument(format!("bad uri '{addr}': {e}")))?
            .connect()
            .await
            .map_err(|e| Status::unavailable(format!("connect '{addr}': {e}")))?;
        cache.insert(addr.to_string(), ch.clone());
        Ok(ch)
    }
}

impl Default for GrpcRingClient {
    fn default() -> Self {
        Self::new()
    }
}

fn to_proto(n: &NodeInfo) -> ProtoNodeInfo {
    ProtoNodeInfo {
        id: n.id,
        address: n.address.clone(),
    }
}

fn from_proto(n: ProtoNodeInfo) -> NodeInfo {
    NodeInfo {
        id: n.id,
        address: n.address,
    }
}

#[async_trait::async_trait]
impl RingClient for GrpcRingClient {
    async fn find_successor(&self, addr: &str, id: u64) -> Result<NodeInfo, Status> {
        let ch = self.channel(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let resp = c.find_successor(FindSuccRequest { id }).await?;
        Ok(from_proto(resp.into_inner()))
    }

    async fn get_predecessor(&self, addr: &str) -> Result<Option<NodeInfo>, Status> {
        let ch = self.channel(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let resp: OptionalNodeInfo = c.get_predecessor(Empty {}).await?.into_inner();
        Ok(resp.node.map(from_proto))
    }

    async fn notify(&self, addr: &str, other: NodeInfo) -> Result<bool, Status> {
        let ch = self.channel(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let resp: BoolMsg = c
            .notify(NotifyRequest {
                other: Some(to_proto(&other)),
            })
            .await?
            .into_inner();
        Ok(resp.ok)
    }

    async fn get_successor_list(&self, addr: &str) -> Result<Vec<NodeInfo>, Status> {
        let ch = self.channel(addr).await?;
        let mut c = GrpcRingClientProto::new(ch);
        let resp: NodeInfoList = c.get_successor_list(Empty {}).await?.into_inner();
        Ok(resp.nodes.into_iter().map(from_proto).collect())
    }

    async fn ping(&self, addr: &str) -> Result<bool, Status> {
        match self.channel(addr).await {
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
}
