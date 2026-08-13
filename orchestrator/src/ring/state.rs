//! SRP: holds ring topology state (successor, predecessor, successor
//! list). No networking, no hashing, no stabilization logic — just
//! guarded mutable state. DIP: stabilization and gRPC layers depend
//! on this abstraction, not vice versa.

use tokio::sync::Mutex;

/// A node's identity in the ring: a u64 ID (hash of address) and the
/// network address other nodes use to reach it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeInfo {
    pub id: u64,
    pub address: String,
}

impl NodeInfo {
    pub fn new(id: u64, address: impl Into<String>) -> Self {
        Self {
            id,
            address: address.into(),
        }
    }
}

/// The mutable ring state for one orchestrator node.
pub struct RingState {
    /// This node's own identity.
    pub self_node: NodeInfo,
    /// Immediate predecessor in the ring (set by `notify`).
    pub predecessor: Mutex<Option<NodeInfo>>,
    /// Immediate successor in the ring (set by stabilization / join).
    pub successor: Mutex<Option<NodeInfo>>,
    /// `r` successors for fault tolerance (Lead successor list).
    pub successor_list: Mutex<Vec<NodeInfo>>,
}

impl RingState {
    pub fn new(self_node: NodeInfo) -> Self {
        Self {
            self_node,
            predecessor: Mutex::new(None),
            successor: Mutex::new(None),
            successor_list: Mutex::new(Vec::new()),
        }
    }

    /// True if `id` falls in the half-open interval (a, b] on the ring,
    /// handling wraparound at u64::MAX -> 0.
    pub fn in_half_open(a: u64, b: u64, id: u64) -> bool {
        if a == b {
            // Empty interval unless it's the full ring (a == b means
            // single-node ring; caller should handle that separately).
            return false;
        }
        if a < b {
            id > a && id <= b
        } else {
            // Wraps around 0.
            id > a || id <= b
        }
    }

    /// True if `id` falls in the open interval (a, b) on the ring.
    pub fn in_open(a: u64, b: u64, id: u64) -> bool {
        if a == b {
            return false;
        }
        if a < b {
            id > a && id < b
        } else {
            id > a || id < b
        }
    }
}
