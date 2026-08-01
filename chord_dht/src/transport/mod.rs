pub mod grpc;

use async_trait::async_trait;

use crate::ring::NodeAddr;

/// Segregated interface for talking to *other* nodes. Keeps transport details
/// out of the core `ChordNode` so the ring logic is testable with a fake
/// transport (DIP + ISP).
#[async_trait]
pub trait RemoteNode: Send + Sync {
    async fn find_successor(&self, addr: &str, id: u64) -> Option<NodeAddr>;
    async fn get_predecessor(&self, addr: &str) -> Option<NodeAddr>;
    async fn notify(&self, addr: &str, self_info: &NodeAddr) -> bool;
    async fn get_local(&self, addr: &str, key: &str) -> Option<String>;
    async fn put_local(&self, addr: &str, key: &str, val: &str) -> bool;
    async fn delete_local(&self, addr: &str, key: &str) -> bool;
    async fn ping(&self, addr: &str) -> bool;
}
