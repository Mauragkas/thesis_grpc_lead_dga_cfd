use super::feature::feature;
use super::leaf::{Anchor, LeafKind, LinearLeaf, RadixSplineLeaf, RADIX_ENTRIES, RP};
use super::model::{PidState, RmiModel};
use super::HASH_SPACE;

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
                pid_state: vec![PidState::default(); bins],
            };
        }
        let leaves = train_linear_leaves(&sorted, bins, n);
        RmiModel {
            stage0_bins: bins,
            leaves,
            n,
            version,
            pid_state: vec![PidState::default(); bins],
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
            let linear = build_model_with_config(&sorted, bins, "linear", version);
            let err = eval_model_error(&linear, &sketch);
            if err < best_error {
                best_error = err;
                best_bins = bins;
                best_kind = "linear".into();
            }
        }

        // Try RadixSpline on the best bin count
        if best_bins >= 16 {
            let radix = build_model_with_config(&sorted, best_bins, "radix", version);
            let err = eval_model_error(&radix, &sketch);
            if err < best_error {
                _ = err;
                best_kind = "radix".into();
            }
        }

        build_model_with_config(&sorted, best_bins, &best_kind, version)
    }
}

// ---------------------------------------------------------------------------
// Helper: build model with given config
// ---------------------------------------------------------------------------
fn build_model_with_config(keys: &[String], bins: usize, kind: &str, version: u64) -> RmiModel {
    let n = keys.len();
    let leaves: Vec<LeafKind> = match kind {
        "radix" => train_radix_leaves(keys, bins, n),
        _ => train_linear_leaves(keys, bins, n),
    };
    RmiModel {
        stage0_bins: bins,
        leaves,
        n,
        version,
        pid_state: vec![PidState::default(); bins],
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
    for bucket in buckets.iter().take(bins) {
        if bucket.len() >= 2 {
            let (w, bias) = linreg(bucket);
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
    let mut buckets: Vec<Vec<(f64, usize)>> = vec![Vec::new(); bins];
    for (i, k) in keys.iter().enumerate() {
        let f = feature(k);
        let b = ((f * bins as f64) as usize).min(bins - 1);
        buckets[b].push((f, i));
    }
    let mut leaves = Vec::with_capacity(bins);
    for bucket in buckets.iter().take(bins) {
        let mut radix_table = vec![0u32; RADIX_ENTRIES];
        if !bucket.is_empty() {
            for (f, rank) in bucket {
                let prefix = ((*f * (RADIX_ENTRIES as f64)) as usize).min(RADIX_ENTRIES - 1);
                let normalized = ((*rank as f64 / n.max(1) as f64) * (u32::MAX as f64)) as u32;
                radix_table[prefix] = normalized;
            }
            // Fill gaps in radix table
            let mut last = 0u32;
            for val in &mut radix_table {
                if *val == 0 {
                    *val = last;
                } else {
                    last = *val;
                }
            }
        }
        leaves.push(LeafKind::RadixSpline(RadixSplineLeaf {
            radix_table,
            rp: RP,
            anchor: Anchor::default(),
        }));
    }
    leaves
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------
fn linreg(points: &[(f64, f64)]) -> (f64, f64) {
    let m = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();

    let denom = m * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return (1.0, 0.0);
    }
    let slope = (m * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / m;
    (slope, intercept)
}

fn eval_model_error(model: &RmiModel, sketch: &[&String]) -> f64 {
    if sketch.is_empty() {
        return 0.0;
    }
    let total_err: f64 = sketch
        .iter()
        .map(|k| {
            let pred = model.predict(k);
            let actual = (feature(k) * HASH_SPACE) as u64;
            (pred as f64 - actual as f64).abs()
        })
        .sum();
    total_err / sketch.len() as f64
}
