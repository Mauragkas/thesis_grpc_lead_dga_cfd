use serde::{Deserialize, Serialize};

use super::leaf::{Anchor, LeafKind, LinearLeaf};
use super::{feature, HASH_SPACE};
use crate::ring::NodeId;

// ---------------------------------------------------------------------------
// PidState
// ---------------------------------------------------------------------------
/// Per-leaf online-learning controller state for the 2-bit PID tuner.
///
/// Replaces the previous bit-packed `u8` (bits 7-6 proportional, 5-2 integral
/// accumulator, 1 derivative sign, 0 derivative magnitude) with named fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PidState {
    /// Proportional state, 0..=3.
    pub proportional: u8,
    /// Quantised integral accumulator, 0..=15.
    pub integral: u8,
    /// Sign of the derivative contribution.
    pub derivative_sign: bool,
    /// Magnitude of the derivative contribution, 0..=1.
    pub derivative_mag: u8,
}

// ---------------------------------------------------------------------------
// RmiModel
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RmiModel {
    pub stage0_bins: usize,
    pub leaves: Vec<LeafKind>,
    pub n: usize,
    pub version: u64,
    /// Per-leaf PID state for online scale updates (not transmitted on the wire).
    #[serde(skip)]
    #[serde(default = "empty_pid_vec")]
    pub pid_state: Vec<PidState>,
}

pub fn empty_pid_vec() -> Vec<PidState> {
    vec![]
}

impl Default for RmiModel {
    fn default() -> Self {
        let bins = 16;
        Self {
            stage0_bins: bins,
            leaves: vec![
                LeafKind::Linear(LinearLeaf {
                    weight: 1.0,
                    bias: 0.0,
                    anchor: Anchor::default(),
                });
                bins
            ],
            n: 0,
            version: 1,
            pid_state: vec![PidState::default(); bins],
        }
    }
}

impl RmiModel {
    /// Predict the LearnedHASH for a key.
    /// Formula:  ⌊(H/N) · f_leaf(key)⌋  where H = 2^64 and N = |training set|.
    /// The CDF prediction f ∈ [0,1] is multiplied by H to recover the hash.
    pub fn predict(&self, key: &str) -> NodeId {
        let f = feature(key);
        let bin = ((f * self.stage0_bins as f64) as usize).min(self.stage0_bins.saturating_sub(1));
        let cdf = self.leaves[bin].predict(f);
        // LEAD formula: floor(H · cdf), but H = HASH_SPACE = 2^64.
        // The N-based scaling is implicit in the training (CDF = rank/N).
        (cdf * HASH_SPACE) as u64
    }
}
