use super::feature::feature;
use super::leaf::{Anchor, LeafKind, LinearLeaf, RadixSplineLeaf, RADIX_ENTRIES, RP};
use super::model::RmiModel;
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
    let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); bins];
    for (i, k) in keys.iter().enumerate() {
        let f = feature(k);
        let p = i as f64 / n.max(1) as f64;
        let b = ((f * bins as f64) as usize).min(bins - 1);
        buckets[b].push((f, p));
    }
    let mut leaves = Vec::with_capacity(bins);
    for bucket in buckets.iter().take(bins) {
        let mut table = vec![0u32; RADIX_ENTRIES];
        if !bucket.is_empty() {
            // Build CDF: for each radix prefix, store the maximum rank
            let mut prefix_max: Vec<usize> = vec![0; RADIX_ENTRIES];
            for (f, _p) in bucket {
                let idx = ((*f * RADIX_ENTRIES as f64) as usize).min(RADIX_ENTRIES - 1);
                prefix_max[idx] = prefix_max[idx].max(bucket.len());
            }
            // Fill: propagate max forward so table is monotonic
            let mut running = 0usize;
            for i in 0..RADIX_ENTRIES {
                running = running.max(prefix_max[i]);
                let cdf = running as f64 / bucket.len().max(1) as f64;
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
