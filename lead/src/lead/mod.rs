pub mod learning;
pub mod lookup;
pub mod node;
pub mod query;
pub mod routing;
pub mod vnode;

pub use node::LeadNode;
pub use vnode::{RmiState, VirtualNode};

pub const R: usize = 4; // successor-list length
pub const DEFAULT_K: usize = 10; // virtual nodes per physical node
pub const DRIFT_THRESHOLD: f64 = 0.40;
pub const MIN_KEYS_FOR_DRIFT: usize = 50;
pub const FRM_GRACE_PERIOD_SECS: u64 = 10;

// Shadow Balancer pruning thresholds
pub const PRUNE_ERROR_RATE: f64 = 0.30;
pub const PRUNE_INACTIVE_SECS: u64 = 120;
pub const PID_ADJUST_INTERVAL: usize = 100;
