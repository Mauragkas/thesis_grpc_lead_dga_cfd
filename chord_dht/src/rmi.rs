use serde::{Deserialize, Serialize};

use crate::ring::NodeId;

const HASH_SPACE: f64 = (u64::MAX as f64) + 1.0;

// ---------------------------------------------------------------------------
// Anchor
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// LinearLeaf
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LinearLeaf {
    pub weight: f64,
    pub bias: f64,
    pub anchor: Anchor,
}

// ---------------------------------------------------------------------------
// RadixSpline — 32-way radix table for CDF prediction
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RadixSplineLeaf {
    pub radix_table: Vec<u32>, // 2^RP entries mapping prefix → rank
    pub rp: usize,             // radix prefix bits (default 10 → 1024 entries)
    pub anchor: Anchor,
}
pub const RP: usize = 10;
pub const RADIX_ENTRIES: usize = 1 << RP;

// ---------------------------------------------------------------------------
// LeafKind enum
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum LeafKind {
    Linear(LinearLeaf),
    RadixSpline(RadixSplineLeaf),
}

impl LeafKind {
    pub fn anchor(&self) -> &Anchor {
        match self {
            LeafKind::Linear(l) => &l.anchor,
            LeafKind::RadixSpline(r) => &r.anchor,
        }
    }

    pub fn anchor_mut(&mut self) -> &mut Anchor {
        match self {
            LeafKind::Linear(l) => &mut l.anchor,
            LeafKind::RadixSpline(r) => &mut r.anchor,
        }
    }

    pub fn predict(&self, feature: f64) -> f64 {
        let raw = match self {
            LeafKind::Linear(l) => l.weight * feature + l.bias,
            LeafKind::RadixSpline(r) => {
                let prefix = (feature * (RADIX_ENTRIES as f64)) as usize;
                let idx = prefix.min(RADIX_ENTRIES - 1);
                (r.radix_table[idx] as f64) / (u32::MAX as f64)
            }
        };
        let anchor = self.anchor();
        let y = raw * anchor.scale + anchor.offset;
        if !y.is_finite() {
            0.0
        } else {
            y.clamp(0.0, 1.0)
        }
    }
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
    /// Per-leaf PID state for online scale updates.  Each entry is a u8
    /// where bits 7-6 = proportional state (2 bits), bits 5-2 = integral
    /// accumulator (4× quantised), bits 1-0 = derivative sign + magnitude.
    /// This implements the 2-bit PID controller specified by LEAD.
    #[serde(skip)]
    #[serde(default = "empty_pid_vec")]
    pub pid_state: Vec<u8>,
}

fn empty_pid_vec() -> Vec<u8> {
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

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------
pub fn feature(key: &str) -> f64 {
    if let Some(hex_part) = key.split('|').next() {
        if !hex_part.is_empty() && hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
            // Normalize to 64-bit regardless of actual hex length.
            // Treat the hex string as a fixed-point fraction: value / 16^len,
            // then scale to u64 range. This makes short prefixes fill the
            // entire [0,1] range proportionally.
            let len = hex_part.len().min(16);
            let prefix = &hex_part[..len];
            if let Ok(val) = u64::from_str_radix(prefix, 16) {
                // Maximum possible value for this length: 16^len - 1
                let max_for_len = if len < 16 {
                    (1u64 << (4 * len)) - 1
                } else {
                    u64::MAX
                };
                if max_for_len > 0 {
                    return val as f64 / max_for_len as f64;
                }
            }
        }
    }
    // Fallback for non-Hilbert keys
    let h = crate::ring::peer_hash(key);
    (h as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------
impl RmiModel {
    /// Train on a sorted key set.
    pub fn train(keys: &[String], version: u64) -> RmiModel {
        let mut sorted: Vec<String> = keys.to_vec();
        sorted.sort();
        sorted.dedup();
        let n = sorted.len();
        let bins = 1;
        if n == 0 {
            return RmiModel {
                stage0_bins: bins,
                leaves: vec![],
                n: 0,
                version,
                pid_state: vec![0u8; bins],
            };
        }
        let leaves = train_linear_leaves(&sorted, bins, n);
        RmiModel {
            stage0_bins: bins,
            leaves,
            n,
            version,
            pid_state: vec![0u8; bins],
        }
    }

    /// Auto-model selection: mountain-climbing on a 1% sketch to pick
    /// the best leaf model type + bin count.
    pub fn train_auto(keys: &[String], version: u64) -> RmiModel {
        let mut sorted: Vec<String> = keys.to_vec();
        sorted.sort();
        sorted.dedup();
        let n = sorted.len();
        if n < 100 {
            return Self::train(keys, version);
        }

        // 1% sketch for evaluation
        let sketch_step = (n / 100).max(1);
        let sketch: Vec<&String> = sorted.iter().step_by(sketch_step).collect();

        // Candidate configs to try
        let bin_counts = [8usize, 16, 32, 64];
        let mut best_error = f64::MAX;
        let mut best_bins = 16usize;
        let mut best_kind: String = "linear".into();

        // Mountain climbing: evaluate each bin count with linear,
        // then try radix on the best bin count.
        for &bins in &bin_counts {
            let linear = build_model_with_config(&sorted, bins, "linear");
            let err = eval_model_error(&linear, &sketch);
            if err < best_error {
                best_error = err;
                best_bins = bins;
                best_kind = "linear".into();
            }
        }

        // Try RadixSpline on the best bin count
        if best_bins >= 16 {
            let radix = build_model_with_config(&sorted, best_bins, "radix");
            let err = eval_model_error(&radix, &sketch);
            if err < best_error {
                _ = err;
                best_kind = "radix".into();
            }
        }

        build_model_with_config(&sorted, best_bins, &best_kind)
    }
}

// ---------------------------------------------------------------------------
// Helper: build model with given config
// ---------------------------------------------------------------------------
fn build_model_with_config(keys: &[String], bins: usize, kind: &str) -> RmiModel {
    let n = keys.len();
    let leaves: Vec<LeafKind> = match kind {
        "radix" => train_radix_leaves(keys, bins, n),
        _ => train_linear_leaves(keys, bins, n),
    };
    RmiModel {
        stage0_bins: bins,
        leaves,
        n,
        version: 1,
        pid_state: vec![0u8; bins],
    }
}

// ---------------------------------------------------------------------------
// Linear leaf training
// ---------------------------------------------------------------------------
fn train_linear_leaves(keys: &[String], bins: usize, n: usize) -> Vec<LeafKind> {
    let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); bins];
    for (i, k) in keys.iter().enumerate() {
        let f = feature(k);
        let p = i as f64 / n.max(1) as f64;
        let b = ((f * bins as f64) as usize).min(bins - 1);
        buckets[b].push((f, p));
    }
    let mut leaves = Vec::with_capacity(bins);
    for b in 0..bins {
        if buckets[b].len() >= 2 {
            let (w, bias) = linreg(&buckets[b]);
            leaves.push(LeafKind::Linear(LinearLeaf {
                weight: w,
                bias,
                anchor: Anchor::default(),
            }));
        } else {
            leaves.push(LeafKind::Linear(LinearLeaf {
                weight: 1.0,
                bias: 0.0,
                anchor: Anchor::default(),
            }));
        }
    }
    leaves
}

// ---------------------------------------------------------------------------
// RadixSpline leaf training
// ---------------------------------------------------------------------------
fn train_radix_leaves(keys: &[String], bins: usize, n: usize) -> Vec<LeafKind> {
    let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); bins];
    for (i, k) in keys.iter().enumerate() {
        let f = feature(k);
        let p = i as f64 / n.max(1) as f64;
        let b = ((f * bins as f64) as usize).min(bins - 1);
        buckets[b].push((f, p));
    }
    let mut leaves = Vec::with_capacity(bins);
    for b in 0..bins {
        let mut table = vec![0u32; RADIX_ENTRIES];
        if !buckets[b].is_empty() {
            // Build CDF: for each radix prefix, store the maximum rank
            let mut prefix_max: Vec<usize> = vec![0; RADIX_ENTRIES];
            for (f, _p) in &buckets[b] {
                let idx = ((*f * RADIX_ENTRIES as f64) as usize).min(RADIX_ENTRIES - 1);
                prefix_max[idx] = prefix_max[idx].max(buckets[b].len());
            }
            // Fill: propagate max forward so table is monotonic
            let mut running = 0usize;
            for i in 0..RADIX_ENTRIES {
                running = running.max(prefix_max[i]);
                let cdf = running as f64 / buckets[b].len().max(1) as f64;
                table[i] = (cdf * u32::MAX as f64) as u32;
            }
        }
        leaves.push(LeafKind::RadixSpline(RadixSplineLeaf {
            radix_table: table,
            rp: RP,
            anchor: Anchor::default(),
        }));
    }
    leaves
}

// ---------------------------------------------------------------------------
// Model error evaluation (mean absolute error in rank prediction)
// ---------------------------------------------------------------------------
fn eval_model_error(model: &RmiModel, sketch: &[&String]) -> f64 {
    let n = sketch.len().max(1) as f64;
    let mut total_err = 0.0;
    for (i, k) in sketch.iter().enumerate() {
        let predicted_hash = model.predict(k) as f64;
        let predicted_rank = predicted_hash / HASH_SPACE * n;
        let actual_rank = i as f64;
        total_err += (predicted_rank - actual_rank).abs();
    }
    total_err / n
}

// ---------------------------------------------------------------------------
// Linear regression
// ---------------------------------------------------------------------------
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
    (w, my - w * mx)
}

// ---------------------------------------------------------------------------
// Differential leaf sync
// -------------------------------------------------------------------------
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

// Proto-compatible diff struct (mirrors the proto message)
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LeafDiff {
    pub index: u32,
    pub weight: f64,
    pub bias: f64,
    pub anchor_offset: f64,
}

fn leaf_params(leaf: &LeafKind) -> (f64, f64, f64) {
    match leaf {
        LeafKind::Linear(l) => (l.weight, l.bias, l.anchor.offset),
        LeafKind::RadixSpline(_) => (0.0, 0.0, leaf.anchor().offset),
    }
}
