use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

pub type NodeId = u64;
pub const M: usize = 64; // 64-bit ring

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct NodeAddr {
    pub id: NodeId,
    pub address: String,
}

pub fn hash(val: &str) -> NodeId {
    let mut hasher = Sha1::new();
    hasher.update(val.as_bytes());
    let result = hasher.finalize();
    u64::from_be_bytes(result[..8].try_into().unwrap())
}

pub fn finger_start(id: NodeId, i: usize) -> NodeId {
    id.wrapping_add(1u64 << i)
}

/// Is `val` in `(start, end]` (inclusive_end=true) or `(start, end)` (false)?
pub fn in_range(val: NodeId, start: NodeId, end: NodeId, inclusive_end: bool) -> bool {
    if start == end {
        // (x, x] wraps the whole ring; (x, x) is empty.
        return inclusive_end;
    }
    if start < end {
        if inclusive_end {
            val > start && val <= end
        } else {
            val > start && val < end
        }
    } else if inclusive_end {
        val > start || val <= end
    } else {
        val > start || val < end
    }
}
