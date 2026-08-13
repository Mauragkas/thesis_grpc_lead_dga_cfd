use crate::ring::in_range;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::LeadNode;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    /// Run the 2-bit PID controller on each leaf's anchor to keep 95% of
    /// keys within each vnode's VID window.
    pub(super) async fn run_pid_adjustment(&self) {
        let snap = self.storage.snapshot().await;
        if snap.is_empty() {
            return;
        }

        let mut rmi = self.rmi.write().await;
        let model = &mut rmi.active;

        // Per-leaf: count keys in their vnode's window
        let bins = model.stage0_bins;
        let mut in_window = vec![0usize; bins];
        let mut outside = vec![0usize; bins];

        for (k, _) in &snap {
            let f = crate::rmi::feature(k);
            let bin = ((f * bins as f64) as usize).min(bins.saturating_sub(1));
            let hash = model.predict(k);
            // Check if hash falls within any vnode's (pred, vid] range
            let mut owned = false;
            for vnode in &self.vnodes {
                let pred = vnode.predecessor.read().await;
                if let Some(p) = pred.as_ref() {
                    if in_range(hash, p.id, vnode.vid, true) {
                        owned = true;
                        break;
                    }
                }
            }
            if owned {
                in_window[bin] += 1;
            } else {
                outside[bin] += 1;
            }
        }

        for b in 0..bins {
            model.adjust_anchor_pid(b, in_window[b], outside[b], 0.95);
        }
        tracing::debug!("PID adjustment complete across {} bins", bins);
    }
}
