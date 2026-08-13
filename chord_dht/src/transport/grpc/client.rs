use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tonic::transport::Channel;

use crate::ring::NodeAddr;
use crate::transport::{RangeResult, RemoteNode};

use super::gen::chord_client::ChordClient;
use super::gen::{
    DeliverRangeRequest, Empty, FindSuccRequest, GetPredRequest, HeartbeatMsg, KeyMsg, ModelParams,
    ModelRequest, NodeAddr as ProtoNode, NotifyRequest, PruneRequest, PutRequest, RangeEntry,
    RangeForwardRequest, RangeRequest, VidMsg,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const RPC_TIMEOUT: Duration = Duration::from_secs(5);
const RANGE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct GrpcRemote {
    channels: Mutex<HashMap<String, Channel>>,
}

impl GrpcRemote {
    pub fn new() -> Self {
        Self {
            channels: Mutex::new(HashMap::new()),
        }
    }

    #[allow(dead_code)]
    async fn deliver_range(
        &self,
        addr: &str,
        entries: &[(String, String)],
        complete: bool,
    ) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.deliver_range(DeliverRangeRequest {
                caller_address: addr.to_string(),
                entries: entries_to_proto(entries),
                complete,
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    #[allow(dead_code)]
    async fn prune_vnode(&self, addr: &str, vid: u64, target_vid: u64, reason: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.prune_vnode(PruneRequest {
                vid,
                target_vid,
                reason: reason.to_string(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    /// Cached channel lookup — connects once per address, reuses thereafter.
    async fn client(&self, addr: &str) -> Option<ChordClient<Channel>> {
        {
            let cache = self.channels.lock().await;
            if let Some(ch) = cache.get(addr) {
                return Some(ChordClient::new(ch.clone()));
            }
        }
        let endpoint = Channel::from_shared(addr.to_string()).ok()?;
        let ch = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect())
            .await
            .ok()?
            .ok()?;
        self.channels
            .lock()
            .await
            .insert(addr.to_string(), ch.clone());
        Some(ChordClient::new(ch))
    }
}

impl Default for GrpcRemote {
    fn default() -> Self {
        Self::new()
    }
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

fn entries_to_proto(v: &[(String, String)]) -> Vec<RangeEntry> {
    v.iter()
        .map(|(k, val)| RangeEntry {
            key: k.clone(),
            value: val.clone(),
        })
        .collect()
}

fn entries_from_proto(v: Vec<RangeEntry>) -> Vec<(String, String)> {
    v.into_iter().map(|e| (e.key, e.value)).collect()
}

#[async_trait]
impl RemoteNode for GrpcRemote {
    async fn deliver_range(
        &self,
        addr: &str,
        entries: &[(String, String)],
        complete: bool,
    ) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.deliver_range(DeliverRangeRequest {
                caller_address: addr.to_string(),
                entries: entries_to_proto(entries),
                complete,
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn prune_vnode(&self, addr: &str, vid: u64, target_vid: u64, reason: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.prune_vnode(PruneRequest {
                vid,
                target_vid,
                reason: reason.to_string(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn get_keys(&self, addr: &str) -> Option<Vec<String>> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.get_keys(Empty {}))
            .await
            .ok()?
            .ok()?;
        Some(resp.into_inner().keys)
    }

    async fn find_successor(&self, addr: &str, vid: u64, id: u64) -> Option<NodeAddr> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.find_successor(FindSuccRequest { vid, id }))
            .await
            .ok()?
            .ok()?;
        Some(from_proto(resp.into_inner()))
    }

    async fn get_predecessor(&self, addr: &str, vid: u64) -> Option<NodeAddr> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.get_predecessor(GetPredRequest { vid }))
            .await
            .ok()?
            .ok()?;
        resp.into_inner().node.map(from_proto)
    }

    async fn get_successor(&self, addr: &str, vid: u64) -> Option<NodeAddr> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.get_successor(VidMsg { vid }))
            .await
            .ok()?
            .ok()?;
        Some(from_proto(resp.into_inner()))
    }

    async fn get_successor_list(&self, addr: &str, vid: u64) -> Vec<NodeAddr> {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return Vec::new(),
        };
        match tokio::time::timeout(RPC_TIMEOUT, c.get_successor_list(VidMsg { vid })).await {
            Ok(Ok(r)) => r.into_inner().nodes.into_iter().map(from_proto).collect(),
            _ => Vec::new(),
        }
    }

    async fn notify(&self, addr: &str, vid: u64, self_info: &NodeAddr) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.notify(NotifyRequest {
                vid,
                other: Some(to_proto(self_info)),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn get_local(&self, addr: &str, key: &str) -> Option<String> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RPC_TIMEOUT,
            c.get_local(KeyMsg {
                key: key.to_string(),
            }),
        )
        .await
        .ok()?
        .ok()?;
        Some(resp.into_inner().value)
    }

    async fn put_local(&self, addr: &str, key: &str, val: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.put_local(PutRequest {
                key: key.to_string(),
                value: val.to_string(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn delete_local(&self, addr: &str, key: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.delete_local(KeyMsg {
                key: key.to_string(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn ping(&self, addr: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(RPC_TIMEOUT, c.ping(Empty {}))
            .await
            .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
            .unwrap_or(false)
    }

    async fn range_query(
        &self,
        addr: &str,
        start_key: &str,
        count: u64,
        caller: &str,
        model_version: u64,
    ) -> Option<RangeResult> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RANGE_TIMEOUT,
            c.range_query(RangeRequest {
                start_key: start_key.to_string(),
                count,
                caller_address: caller.to_string(),
                model_version,
            }),
        )
        .await
        .ok()?
        .ok()?;
        let inner = resp.into_inner();
        Some(RangeResult {
            entries: entries_from_proto(inner.entries),
            complete: inner.complete,
            next_address: inner.next_address,
        })
    }

    async fn range_forward(
        &self,
        addr: &str,
        from_key: &str,
        count: u64,
        caller: &str,
        origin_vid: u64,
        model_version: u64,
        payload: Vec<(String, String)>,
    ) -> Option<RangeResult> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RANGE_TIMEOUT,
            c.range_forward(RangeForwardRequest {
                start_key: from_key.to_string(),
                count,
                caller_address: caller.to_string(),
                payload: entries_to_proto(&payload),
                origin_vid,
                model_version,
            }),
        )
        .await
        .ok()?
        .ok()?;
        let inner = resp.into_inner();
        Some(RangeResult {
            entries: entries_from_proto(inner.entries),
            complete: inner.complete,
            next_address: inner.next_address,
        })
    }

    async fn push_model(&self, addr: &str, version: u64, data: &[u8]) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.push_model(ModelParams {
                version,
                data: data.to_vec(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn request_model(&self, addr: &str, coordinator: &str) -> Option<(u64, Vec<u8>)> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RPC_TIMEOUT,
            c.request_model(ModelRequest {
                coordinator: coordinator.to_string(),
            }),
        )
        .await
        .ok()?
        .ok()?;
        let inner = resp.into_inner();
        Some((inner.version, inner.data))
    }

    async fn heartbeat(
        &self,
        addr: &str,
        sender: &str,
        update_ready: bool,
        model_version: u64,
    ) -> Option<bool> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RPC_TIMEOUT,
            c.heartbeat(HeartbeatMsg {
                sender: sender.to_string(),
                update_ready,
                model_version,
            }),
        )
        .await
        .ok()?
        .ok()?;
        Some(resp.into_inner().ok)
    }
}
