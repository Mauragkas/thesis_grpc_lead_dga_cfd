use serde::{Deserialize, Serialize};

use super::leaf::{Anchor, LeafKind, LinearLeaf};
use super::{feature, HASH_SPACE};
use crate::ring::NodeId;

// ---------------------------------------------------------------------------
// RmiModel
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RmiModel {
    pub stage0_bins: usize,
    pub leaves: Vec<LeafKind>,
    pub n: usize,
    pub version: u64,
    /// Per-leaf PID state for online scale updates.  Each entry is a u8
    /// where bits 7-6 = proportional state (2 bits), bits 5-2 = integral
    /// accumulator (4× quantised), bits 1-0 = derivative sign + magnitude.
    /// This implements the 2-bit PID controller specified by LEAD.
    #[serde(skip)]
    #[serde(default = "empty_pid_vec")]
    pub pid_state: Vec<u8>,
}

pub fn empty_pid_vec() -> Vec<u8> {
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
            pid_state: vec![0u8; bins],
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

    // -----------------------------------------------------------------------
    // Online anchor adjustment with 2-bit PID
    // -----------------------------------------------------------------------
    /// Called periodically (every ~handful of insertions) to nudge the anchor
    /// so that the 95% key quantile stays inside this vnode's VID window.
    ///
    /// `keys_in_window`  = count of local keys whose hash falls in (pred, vid].
    /// `keys_outside`    = count outside.
    /// `leaf_idx`        = which leaf to adjust.
    /// `target_ratio`    = desired fraction inside (e.g. 0.95).
    pub fn adjust_anchor_pid(
        &mut self,
        leaf_idx: usize,
        keys_in_window: usize,
        keys_outside: usize,
        target_ratio: f64,
    ) {
        if leaf_idx >= self.pid_state.len() {
            self.pid_state.resize(self.stage0_bins, 0u8);
        }
        let total = keys_in_window.saturating_add(keys_outside);
        if total < 20 {
            return;
        } // not enough data

        let actual = keys_in_window as f64 / total.max(1) as f64;
        let error = actual - target_ratio;

        let state = &mut self.pid_state[leaf_idx];
        let p_state = (*state >> 6) & 0b11; // 2-bit proportional
        let i_acc = (*state >> 2) & 0b1111; // 4-bit integral accumulator
        let d_sign = (*state & 0b10) != 0; // derivative sign
        let d_mag = *state & 0b01;

        // map error to new proportional state
        let new_p = if error > 0.05 {
            3u8
        } else if error > 0.02 {
            2u8
        } else if error > -0.02 {
            1u8
        } else {
            0u8
        };

        // 2-bit PID output = clamp(p_state + i_acc/4 + d)
        let i_contrib = i_acc as f64 / 4.0;
        let d_contrib = if d_sign {
            d_mag as f64
        } else {
            -(d_mag as f64)
        };
        let pid = (new_p as f64) + i_contrib + d_contrib;

        // scale adjustment: discrete steps
        let step = 0.05;
        let anchor = self.leaves[leaf_idx].anchor_mut();
        if pid > 2.0 {
            anchor.scale *= 1.0 + step;
            // reset integral on large correction
            *state = (new_p << 6) | (1u8 << 2);
        } else if pid < 0.5 {
            anchor.scale *= 1.0 - step;
            *state = (new_p << 6) | (1u8 << 2);
        } else {
            // accumulate integral
            let new_i = (i_acc.saturating_add(1)).min(15);
            // derivative = sign(prev_error) * magnitude
            let prev_p = p_state as i8;
            let curr_p = new_p as i8;
            let d = if curr_p > prev_p {
                1u8
            } else if curr_p < prev_p {
                2u8
            }
            // d_sign=true, d_mag=0? no...
            else {
                0u8
            };
            *state = (new_p << 6) | (new_i << 2) | d;
        }

        // offset centering: shift to keep median inside window
        let centering_step = 0.01;
        if error > 0.0 {
            anchor.offset -= centering_step * error.abs().min(1.0);
        } else {
            anchor.offset += centering_step * error.abs().min(1.0);
        }
    }
}
