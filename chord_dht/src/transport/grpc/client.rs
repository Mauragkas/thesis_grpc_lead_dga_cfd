use async_trait::async_trait;
use tonic::transport::Channel;

use crate::ring::NodeAddr;
use crate::transport::RemoteNode;

use super::gen::chord_client::ChordClient;
use super::gen::{Empty, KeyMsg, NodeAddr as ProtoNode, NodeIdMsg, PutRequest};

/// gRPC transport implementing the `RemoteNode` abstraction (OCP: a new
/// transport added without modifying `ChordNode`).
///
/// Addresses are expected as `http://host:port` (tonic's URI scheme).
pub struct GrpcRemote;

impl GrpcRemote {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GrpcRemote {
    fn default() -> Self {
        Self::new()
    }
}

async fn connect(addr: &str) -> Option<ChordClient<Channel>> {
    ChordClient::connect(addr.to_string()).await.ok()
}

fn to_proto(n: &NodeAddr) -> ProtoNode {
    ProtoNode {
        id: n.id,
        address: n.address.clone(),
    }
}

fn from_proto(n: ProtoNode) -> NodeAddr {
    NodeAddr {
        id: n.id,
        address: n.address,
    }
}

#[async_trait]
impl RemoteNode for GrpcRemote {
    async fn find_successor(&self, addr: &str, id: u64) -> Option<NodeAddr> {
        let mut c = connect(addr).await?;
        let resp = c.find_successor(NodeIdMsg { id }).await.ok()?;
        Some(from_proto(resp.into_inner()))
    }

    async fn get_successor(&self, addr: &str) -> Option<NodeAddr> {
        let mut c = connect(addr).await?;
        let resp = c.get_successor(Empty {}).await.ok()?;
        Some(from_proto(resp.into_inner()))
    }

    async fn get_successor_list(&self, addr: &str) -> Vec<NodeAddr> {
        let mut c = match connect(addr).await {
            Some(c) => c,
            None => return Vec::new(),
        };
        match c.get_successor_list(Empty {}).await {
            Ok(r) => r.into_inner().nodes.into_iter().map(from_proto).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn get_predecessor(&self, addr: &str) -> Option<NodeAddr> {
        let mut c = connect(addr).await?;
        let resp = c.get_predecessor(Empty {}).await.ok()?;
        resp.into_inner().node.map(from_proto)
    }

    async fn notify(&self, addr: &str, self_info: &NodeAddr) -> bool {
        let mut c = match connect(addr).await {
            Some(c) => c,
            None => return false,
        };
        c.notify(to_proto(self_info))
            .await
            .map(|r| r.into_inner().ok)
            .unwrap_or(false)
    }

    async fn get_local(&self, addr: &str, key: &str) -> Option<String> {
        let mut c = connect(addr).await?;
        let resp = c
            .get_local(KeyMsg {
                key: key.to_string(),
            })
            .await
            .ok()?;
        let v = resp.into_inner().value;
        Some(v)
    }

    async fn put_local(&self, addr: &str, key: &str, val: &str) -> bool {
        let mut c = match connect(addr).await {
            Some(c) => c,
            None => return false,
        };
        c.put_local(PutRequest {
            key: key.to_string(),
            value: val.to_string(),
        })
        .await
        .map(|r| r.into_inner().ok)
        .unwrap_or(false)
    }

    async fn delete_local(&self, addr: &str, key: &str) -> bool {
        let mut c = match connect(addr).await {
            Some(c) => c,
            None => return false,
        };
        c.delete_local(KeyMsg {
            key: key.to_string(),
        })
        .await
        .map(|r| r.into_inner().ok)
        .unwrap_or(false)
    }

    async fn ping(&self, addr: &str) -> bool {
        let mut c = match connect(addr).await {
            Some(c) => c,
            None => return false,
        };
        match c.ping(Empty {}).await {
            Ok(r) => r.into_inner().ok,
            Err(_) => false,
        }
    }
}
