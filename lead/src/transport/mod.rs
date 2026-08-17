pub mod grpc;

use async_trait::async_trait;

use crate::ring::NodeAddr;

#[derive(Clone, Debug, Default)]
pub struct RangeResult {
    pub entries: Vec<(String, String)>,
    pub complete: bool,
    pub next_address: String,
}

/// Ring-membership RPCs (successor/predecessor/finger maintenance).
#[async_trait]
pub trait RingClient: Send + Sync {
    async fn find_successor(&self, addr: &str, vid: u64, id: u64) -> Option<NodeAddr>;
    async fn get_predecessor(&self, addr: &str, vid: u64) -> Option<NodeAddr>;
    async fn get_successor(&self, addr: &str, vid: u64) -> Option<NodeAddr>;
    async fn get_successor_list(&self, addr: &str, vid: u64) -> Vec<NodeAddr>;
    async fn notify(&self, addr: &str, vid: u64, self_info: &NodeAddr) -> bool;
    async fn ping(&self, addr: &str) -> bool;
}

/// Key-value storage RPCs.
#[async_trait]
pub trait KvClient: Send + Sync {
    async fn get_local(&self, addr: &str, key: &str) -> Option<String>;
    async fn put_local(&self, addr: &str, key: &str, val: &str) -> bool;
    async fn delete_local(&self, addr: &str, key: &str) -> bool;
    async fn get_keys(&self, addr: &str) -> Option<Vec<String>>;
}

/// Range-scan RPCs (learned-hash range queries).
#[async_trait]
pub trait RangeClient: Send + Sync {
    async fn range_query(
        &self,
        addr: &str,
        start_key: &str,
        count: u64,
        caller: &str,
        model_version: u64,
    ) -> Option<RangeResult>;
    async fn range_forward(
        &self,
        addr: &str,
        from_key: &str,
        count: u64,
        caller: &str,
        origin_vid: u64,
        model_version: u64,
        payload: Vec<(String, String)>,
    ) -> Option<RangeResult>;
}

/// Federated-model RPCs (push/request/heartbeat).
#[async_trait]
pub trait ModelClient: Send + Sync {
    async fn push_model(&self, addr: &str, version: u64, data: &[u8]) -> bool;
    async fn request_model(&self, addr: &str, coordinator: &str) -> Option<(u64, Vec<u8>)>;
    async fn heartbeat(
        &self,
        addr: &str,
        sender: &str,
        update_ready: bool,
        model_version: u64,
    ) -> Option<bool>;
}

/// Marker combining the four client concerns. Keeps existing `R: RemoteNode`
/// bounds working while exposing narrow per-concern traits (ISP).
pub trait RemoteNode: RingClient + KvClient + RangeClient + ModelClient + Send + Sync {}
impl<T: RingClient + KvClient + RangeClient + ModelClient + Send + Sync> RemoteNode for T {}
