pub mod diff;
pub mod feature;
pub mod leaf;
pub mod model;
pub mod train;

pub use diff::LeafDiff;
pub use feature::feature;
pub use leaf::{Anchor, LeafKind, LinearLeaf, RadixSplineLeaf, RADIX_ENTRIES, RP};
pub use model::{empty_pid_vec, PidState, RmiModel};

/// 64-bit hash space (matches ring::M).
pub const HASH_SPACE: f64 = (u64::MAX as f64) + 1.0;
