//! ISP: the narrow set of remote operations the ring needs from other
//! nodes. Stabilization and join depend on this trait (DIP), never on
//! the concrete gRPC client.

use crate::ring::state::NodeInfo;
use tonic::Status;

#[async_trait::async_trait]
pub trait RingClient: Send + Sync {
    /// Ask the node at `addr` to find the successor of `id`.
    async fn find_successor(&self, addr: &str, id: u64) -> Result<NodeInfo, Status>;

    /// Ask the node at `addr` for its current predecessor.
    async fn get_predecessor(&self, addr: &str) -> Result<Option<NodeInfo>, Status>;

    /// Notify the node at `addr` that `other` thinks it is its predecessor.
    async fn notify(&self, addr: &str, other: NodeInfo) -> Result<bool, Status>;

    /// Ask the node at `addr` for its successor list (fault-tolerance).
    async fn get_successor_list(&self, addr: &str) -> Result<Vec<NodeInfo>, Status>;

    /// Liveness check.
    async fn ping(&self, addr: &str) -> Result<bool, Status>;
}
