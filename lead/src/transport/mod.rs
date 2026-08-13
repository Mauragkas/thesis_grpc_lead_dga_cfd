pub mod grpc;

use async_trait::async_trait;

use crate::ring::NodeAddr;

#[derive(Clone, Debug, Default)]
pub struct RangeResult {
    pub entries: Vec<(String, String)>,
    pub complete: bool,
    pub next_address: String,
}

#[async_trait]
pub trait RemoteNode: Send + Sync {
    async fn find_successor(&self, addr: &str, vid: u64, id: u64) -> Option<NodeAddr>;
    async fn get_predecessor(&self, addr: &str, vid: u64) -> Option<NodeAddr>;
    async fn get_successor(&self, addr: &str, vid: u64) -> Option<NodeAddr>;
    async fn get_successor_list(&self, addr: &str, vid: u64) -> Vec<NodeAddr>;
    async fn notify(&self, addr: &str, vid: u64, self_info: &NodeAddr) -> bool;
    async fn get_local(&self, addr: &str, key: &str) -> Option<String>;
    async fn put_local(&self, addr: &str, key: &str, val: &str) -> bool;
    async fn delete_local(&self, addr: &str, key: &str) -> bool;
    async fn ping(&self, addr: &str) -> bool;
    async fn get_keys(&self, addr: &str) -> Option<Vec<String>>;

    async fn range_query(
        &self,
        addr: &str,
        start_key: &str,
        count: u64,
        caller: &str,
        model_version: u64, // NEW
    ) -> Option<RangeResult>;
    async fn range_forward(
        &self,
        addr: &str,
        from_key: &str,
        count: u64,
        caller: &str,
        origin_vid: u64,
        model_version: u64, // NEW
        payload: Vec<(String, String)>,
    ) -> Option<RangeResult>;

    async fn deliver_range(&self, addr: &str, entries: &[(String, String)], complete: bool)
        -> bool;
    async fn prune_vnode(&self, addr: &str, vid: u64, target_vid: u64, reason: &str) -> bool;

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
