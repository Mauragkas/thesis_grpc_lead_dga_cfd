use serde::{Deserialize, Serialize};

use super::leaf::LeafKind;
use super::model::RmiModel;

// Proto-compatible diff struct (mirrors the proto message)
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LeafDiff {
    pub index: u32,
    pub weight: f64,
    pub bias: f64,
    pub anchor_offset: f64,
}

impl RmiModel {
    /// Return indices + parameters of leaves that differ from `other`.
    /// Used for the ~12 KB differential sync.
    pub fn diff_leaves(&self, other: &RmiModel) -> Vec<(usize, LeafDiff)> {
        let mut diffs = Vec::new();
        let max_bins = self.stage0_bins.max(other.stage0_bins);
        for i in 0..max_bins {
            let s = self.leaves.get(i);
            let o = other.leaves.get(i);
            if s.is_none() || o.is_none() {
                diffs.push((
                    i,
                    LeafDiff {
                        index: i as u32,
                        weight: 0.0,
                        bias: 0.0,
                        anchor_offset: 0.0,
                    },
                ));
                continue;
            }
            let (sw, sb, so_off) = leaf_params(s.unwrap());
            let (ow, ob, oo_off) = leaf_params(o.unwrap());
            if (sw - ow).abs() > 1e-12 || (sb - ob).abs() > 1e-12 || (so_off - oo_off).abs() > 1e-12
            {
                diffs.push((
                    i,
                    LeafDiff {
                        index: i as u32,
                        weight: sw - ow,
                        bias: sb - ob,
                        anchor_offset: so_off - oo_off,
                    },
                ));
            }
        }
        diffs
    }

    /// Apply a set of leaf diffs, returning a new model.
    pub fn apply_diff(&self, diffs: &[LeafDiff], new_version: u64) -> RmiModel {
        let mut model = self.clone();
        model.version = new_version;
        for d in diffs {
            let idx = d.index as usize;
            if idx < model.leaves.len() {
                match &mut model.leaves[idx] {
                    LeafKind::Linear(l) => {
                        l.weight += d.weight;
                        l.bias += d.bias;
                        l.anchor.offset += d.anchor_offset;
                    }
                    LeafKind::RadixSpline(r) => {
                        r.anchor.offset += d.anchor_offset;
                    }
                }
            }
        }
        model
    }
}

fn leaf_params(leaf: &LeafKind) -> (f64, f64, f64) {
    match leaf {
        LeafKind::Linear(l) => (l.weight, l.bias, l.anchor.offset),
        LeafKind::RadixSpline(_) => (0.0, 0.0, leaf.anchor().offset),
    }
}
