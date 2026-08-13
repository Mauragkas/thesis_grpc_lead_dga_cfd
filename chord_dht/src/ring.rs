use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

pub type NodeId = u64;
pub const M: usize = 64; // 64-bit ring
pub const F: usize = 18; // LEAD finger table size (base-10 spacing, 10^17 max)

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct NodeAddr {
    pub id: NodeId,
    pub address: String,
}

/// PeerHASH: uniform cryptographic hash used ONLY for node/virtual-node
/// identity (VIDs). Never used for data-key placement (Step 2 invariant).
pub fn peer_hash(val: &str) -> NodeId {
    let mut hasher = Sha1::new();
    hasher.update(val.as_bytes());
    let result = hasher.finalize();
    u64::from_be_bytes(result[..8].try_into().unwrap())
}

/// LEAD finger spacing: for i=0 the finger is the immediate successor
/// (handled by the Chord protocol directly); for i≥1,
/// finger[i] points to `id + 10^(i-1)`.
/// Last valid index is F-1 (10^16).
pub fn finger_start(id: NodeId, i: usize) -> NodeId {
    if i == 0 {
        // finger[0] = immediate successor — computed by Chord, not by this formula
        return id;
    }
    let exp = (i - 1) as u32;
    let jump = if exp < 19 {
        10u64.saturating_pow(exp)
    } else {
        u64::MAX
    };
    id.wrapping_add(jump)
}

/// Is `val` in `(start, end]` (inclusive_end=true) or `(start, end)` (false)?
pub fn in_range(val: NodeId, start: NodeId, end: NodeId, inclusive_end: bool) -> bool {
    if start == end {
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
