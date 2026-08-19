pub mod learning;
pub mod lookup;
pub mod node;
pub mod query;
pub mod routing;
pub mod vnode;

pub use node::LeadNode;
pub use vnode::{RmiState, VirtualNode};

// Backward-compatible default aliases from centralized Config
pub use crate::config::{
    DEFAULT_DRIFT_THRESHOLD as DRIFT_THRESHOLD,
    DEFAULT_FRM_GRACE_PERIOD_SECS as FRM_GRACE_PERIOD_SECS,
    DEFAULT_MIN_KEYS_FOR_DRIFT as MIN_KEYS_FOR_DRIFT,
    DEFAULT_PID_ADJUST_INTERVAL as PID_ADJUST_INTERVAL,
    DEFAULT_PRUNE_ERROR_RATE as PRUNE_ERROR_RATE,
    DEFAULT_PRUNE_INACTIVE_SECS as PRUNE_INACTIVE_SECS,
    DEFAULT_SUCCESSOR_LIST_LEN as R,
    DEFAULT_VIRTUAL_NODE_COUNT as DEFAULT_K,
};
