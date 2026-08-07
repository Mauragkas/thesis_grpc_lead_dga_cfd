use serde::{Deserialize, Serialize};

use crate::ring::NodeId;

const HASH_SPACE: f64 = (u64::MAX as f64) + 1.0;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Anchor {
    pub offset: f64,
    pub scale: f64,
}

impl Default for Anchor {
    fn default() -> Self {
        Self {
            offset: 0.0,
            scale: 1.0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LinearLeaf {
    pub weight: f64,
    pub bias: f64,
    pub anchor: Anchor,
}

/// 2-stage Recursive Model Index. Stage-0 selects a leaf by feature bucket;
/// the leaf predicts the normalized CDF position p in [0,1]; LearnedHASH =
/// floor(p * H). The model is order-preserving because `feature` is monotonic
/// in the key's leading bytes and the leaf is a (clamped) linear function.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RmiModel {
    pub stage0_bins: usize,
    pub leaves: Vec<LinearLeaf>,
    pub n: usize,
    pub version: u64,
}

impl Default for RmiModel {
    fn default() -> Self {
        let bins = 16;
        Self {
            stage0_bins: bins,
            leaves: vec![
                LinearLeaf {
                    weight: 1.0,
                    bias: 0.0,
                    anchor: Anchor::default()
                };
                bins
            ],
            n: 0,
            version: 1,
        }
    }
}

// Hash the full key (not just first 8 bytes) so configs with shared
// JSON prefixes still get distinct, uniformly distributed features.
fn feature(key: &str) -> f64 {
    // For Hilbert-prefixed keys ("hex|json"), parse the hex prefix as the
    // feature. This makes the feature monotonic in the key's lexicographic
    // order, so learned_hash(key) preserves key order and range queries
    // return keys in the correct Hilbert sequence.
    if let Some(hex_part) = key.split('|').next() {
        if !hex_part.is_empty() && hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
            // Use the first 16 hex chars (64 bits). Since the hex prefix is
            // zero-padded, this gives the most-significant 64 bits of the
            // Hilbert index — sufficient for monotonic vnode placement.
            let prefix = &hex_part[..hex_part.len().min(16)];
            if let Ok(h) = u64::from_str_radix(prefix, 16) {
                return h as f64 / (u64::MAX as f64);
            }
        }
    }
    // Fallback for non-Hilbert keys: uniform hash (original behavior)
    let h = crate::ring::peer_hash(key);
    (h as f64) / (u64::MAX as f64)
}

impl RmiModel {
    pub fn predict(&self, key: &str) -> NodeId {
        let f = feature(key);
        let bin = ((f * self.stage0_bins as f64) as usize).min(self.stage0_bins.saturating_sub(1));
        let leaf = &self.leaves[bin];
        let mut y = (leaf.weight * f + leaf.bias) * leaf.anchor.scale + leaf.anchor.offset;
        if !y.is_finite() {
            y = 0.0;
        }
        let y = y.clamp(0.0, 1.0);
        (y * HASH_SPACE) as u64
    }

    /// Train on a 1%+ sketch of local keys (here: full sorted key set).
    /// Each leaf fits p = w*feature + bias via least squares on its bucket.
    pub fn train(keys: &[String], version: u64) -> RmiModel {
        let mut sorted: Vec<String> = keys.to_vec();
        sorted.sort();
        sorted.dedup();
        let n = sorted.len();
        let bins = 16;
        let mut leaves = vec![
            LinearLeaf {
                weight: 1.0,
                bias: 0.0,
                anchor: Anchor::default()
            };
            bins
        ];
        if n == 0 {
            return RmiModel {
                stage0_bins: bins,
                leaves,
                n,
                version,
            };
        }
        let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); bins];
        for (i, k) in sorted.iter().enumerate() {
            let f = feature(k);
            let p = i as f64 / n as f64; // normalized rank (CDF)
            let b = ((f * bins as f64) as usize).min(bins - 1);
            buckets[b].push((f, p));
        }
        for b in 0..bins {
            if buckets[b].len() >= 2 {
                let (w, bias) = linreg(&buckets[b]);
                leaves[b] = LinearLeaf {
                    weight: w,
                    bias,
                    anchor: Anchor::default(),
                };
            }
        }
        RmiModel {
            stage0_bins: bins,
            leaves,
            n,
            version,
        }
    }
}

/// Numerically stable linear regression using the centered formula.
/// Avoids catastrophic cancellation when features are very close together
/// (e.g., keys sharing a common prefix).
fn linreg(samples: &[(f64, f64)]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mx: f64 = samples.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let my: f64 = samples.iter().map(|(_, y)| *y).sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in samples {
        let dx = x - mx;
        let dy = y - my;
        num += dx * dy;
        den += dx * dx;
    }

    if den.abs() < 1e-30 {
        return (0.0, my);
    }

    let w = num / den;
    let bias = my - w * mx;
    (w, bias)
}
