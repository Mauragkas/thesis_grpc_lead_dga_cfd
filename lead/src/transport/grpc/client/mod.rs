mod convert;
mod kv;
mod model;
mod range;
mod ring;

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tonic::transport::Channel;

use crate::ring::NodeAddr;
use crate::transport::{KvClient, ModelClient, RangeClient, RangeResult, RingClient};

use super::gen::lead_client::LeadClient;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
pub(super) const RPC_TIMEOUT: Duration = Duration::from_secs(5);
pub(super) const RANGE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct GrpcRemote {
    pub(super) channels: Mutex<HashMap<String, Channel>>,
}

impl GrpcRemote {
    pub fn new() -> Self {
        Self {
            channels: Mutex::new(HashMap::new()),
        }
    }

    /// Cached channel lookup — connects once per address, reuses thereafter.
    pub(super) async fn client(&self, addr: &str) -> Option<LeadClient<Channel>> {
        {
            let cache = self.channels.lock().await;
            if let Some(ch) = cache.get(addr) {
                return Some(LeadClient::new(ch.clone()));
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
        Some(LeadClient::new(ch))
    }
}

impl Default for GrpcRemote {
    fn default() -> Self {
        Self::new()
    }
}

// Single, authoritative transport implementation. Bodies live in the
// `kv`, `ring`, `range`, and `model` submodules as inherent helpers.

#[async_trait]
impl KvClient for GrpcRemote {
    async fn get_local(&self, addr: &str, key: &str) -> Option<String> {
        self.get_local_inner(addr, key).await
    }
    async fn put_local(&self, addr: &str, key: &str, val: &str) -> bool {
        self.put_local_inner(addr, key, val).await
    }
    async fn delete_local(&self, addr: &str, key: &str) -> bool {
        self.delete_local_inner(addr, key).await
    }
    async fn get_keys(&self, addr: &str) -> Option<Vec<String>> {
        self.get_keys_inner(addr).await
    }
}

#[async_trait]
impl RingClient for GrpcRemote {
    async fn find_successor(&self, addr: &str, vid: u64, id: u64) -> Option<NodeAddr> {
        self.find_successor_inner(addr, vid, id).await
    }
    async fn get_predecessor(&self, addr: &str, vid: u64) -> Option<NodeAddr> {
        self.get_predecessor_inner(addr, vid).await
    }
    async fn get_successor(&self, addr: &str, vid: u64) -> Option<NodeAddr> {
        self.get_successor_inner(addr, vid).await
    }
    async fn get_successor_list(&self, addr: &str, vid: u64) -> Vec<NodeAddr> {
        self.get_successor_list_inner(addr, vid).await
    }
    async fn notify(&self, addr: &str, vid: u64, self_info: &NodeAddr) -> bool {
        self.notify_inner(addr, vid, self_info).await
    }
    async fn ping(&self, addr: &str) -> bool {
        self.ping_inner(addr).await
    }
}

#[async_trait]
impl RangeClient for GrpcRemote {
    async fn range_query(
        &self,
        addr: &str,
        start_key: &str,
        count: u64,
        caller: &str,
        model_version: u64,
    ) -> Option<RangeResult> {
        self.range_query_inner(addr, start_key, count, caller, model_version)
            .await
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
        self.range_forward_inner(
            addr,
            from_key,
            count,
            caller,
            origin_vid,
            model_version,
            payload,
        )
        .await
    }
}

#[async_trait]
impl ModelClient for GrpcRemote {
    async fn push_model(&self, addr: &str, version: u64, data: &[u8]) -> bool {
        self.push_model_inner(addr, version, data).await
    }
    async fn request_model(&self, addr: &str, coordinator: &str) -> Option<(u64, Vec<u8>)> {
        self.request_model_inner(addr, coordinator).await
    }
    async fn heartbeat(
        &self,
        addr: &str,
        sender: &str,
        update_ready: bool,
        model_version: u64,
    ) -> Option<bool> {
        self.heartbeat_inner(addr, sender, update_ready, model_version)
            .await
    }
}
