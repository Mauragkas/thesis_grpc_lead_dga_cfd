use crate::rmi::{Anchor, LeafKind, LinearLeaf, PidState, RmiModel};

// ------------------------------------------------------------------
// Federated Averaging (unchanged, but handles both leaf kinds)
// ------------------------------------------------------------------
pub fn fed_avg(models: Vec<RmiModel>, version: u64) -> RmiModel {
    if models.is_empty() {
        return RmiModel::default();
    }
    let bins = models[0].stage0_bins;
    let total_n: usize = models.iter().map(|m| m.n.max(1)).sum();
    let mut leaves = vec![
        LeafKind::Linear(LinearLeaf {
            weight: 0.0,
            bias: 0.0,
            anchor: Anchor::default()
        });
        bins
    ];
    for (b, leaf) in leaves.iter_mut().enumerate() {
        let mut w_sum = 0.0;
        let mut bias_sum = 0.0;
        let mut off_sum = 0.0;
        let mut count = 0usize;
        for m in &models {
            let wt = m.n as f64 / total_n as f64;
            if let Some(kind) = m.leaves.get(b) {
                match kind {
                    LeafKind::Linear(l) => {
                        w_sum += l.weight * wt;
                        bias_sum += l.bias * wt;
                        off_sum += l.anchor.offset * wt;
                        count += 1;
                    }
                    LeafKind::RadixSpline(r) => {
                        off_sum += r.anchor.offset * wt;
                        count += 1;
                    }
                }
            }
        }
        if count > 0 {
            *leaf = LeafKind::Linear(LinearLeaf {
                weight: w_sum,
                bias: bias_sum,
                anchor: Anchor {
                    offset: off_sum,
                    scale: 1.0,
                },
            });
        }
    }
    RmiModel {
        stage0_bins: bins,
        leaves,
        n: total_n,
        version,
        pid_state: vec![PidState::default(); bins],
    }
}
