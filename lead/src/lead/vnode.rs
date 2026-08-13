use crate::ring::{NodeAddr, NodeId, F};
use crate::rmi::RmiModel;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::RwLock;

/// Per-virtual-node ring state. Each VID runs an independent Lead protocol.
pub struct VirtualNode {
    pub vid: NodeId,
    pub predecessor: RwLock<Option<NodeAddr>>,
    pub fingers: RwLock<Vec<Option<NodeAddr>>>,
    pub successor_list: RwLock<Vec<NodeAddr>>,

    // Shadow Balancer pruning metrics (Step 2.2)
    pub request_count: AtomicU64,
    pub error_count: AtomicU64,
    pub last_active: RwLock<Instant>,
    pub pruned: AtomicU64, // 0 = active, 1 = pruned
}

impl VirtualNode {
    pub(crate) fn new(vid: NodeId, self_addr: &str) -> Self {
        let self_node = NodeAddr {
            id: vid,
            address: self_addr.to_string(),
        };
        Self {
            vid,
            predecessor: RwLock::new(None),
            fingers: RwLock::new(vec![Some(self_node.clone()); F]),
            successor_list: RwLock::new(vec![self_node]),
            request_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            last_active: RwLock::new(Instant::now()),
            pruned: AtomicU64::new(0),
        }
    }

    pub async fn successor(&self) -> NodeAddr {
        self.successor_list.read().await.first().cloned().unwrap()
    }

    pub fn error_rate(&self) -> f64 {
        let total = self.request_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.error_count.load(Ordering::Relaxed) as f64 / total as f64
    }

    pub fn record_request(&self) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }
    pub async fn mark_active(&self) {
        *self.last_active.write().await = Instant::now();
    }
}

pub struct RmiState {
    pub active: RmiModel,
    pub update: Option<RmiModel>,
    pub drift_new: usize,
    pub update_ready: bool,
    /// Tracks which leaves changed since last broadcast (for differential sync)
    pub dirty_leaves: std::collections::HashSet<usize>,
}
