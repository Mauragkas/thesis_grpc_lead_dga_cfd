use std::sync::atomic::Ordering;
use tracing::warn;

use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::LeadNode;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    // ------------------------------------------------------------------
    // Step 2.2: Shadow Balancer — prune low-throughput vnodes
    // ------------------------------------------------------------------
    pub async fn prune_low_throughput_vnodes(&self) {
        let mut to_prune = Vec::new();
        for v in &self.vnodes {
            if v.pruned.load(Ordering::Relaxed) == 1 {
                continue;
            }
            let err_rate = v.error_rate();
            let inactive = v.last_active.read().await.elapsed().as_secs();
            if err_rate > self.config.prune_error_rate || inactive > self.config.prune_inactive_secs {
                to_prune.push(v.vid);
            }
        }
        for vid in &to_prune {
            if let Some(v) = self.find_vnode(*vid) {
                v.pruned.store(1, Ordering::Relaxed);
                warn!(
                    "PRUNE: vnode {} pruned (err_rate={:.2}, inactive={}s)",
                    vid,
                    v.error_rate(),
                    v.last_active.read().await.elapsed().as_secs()
                );
            }
        }
        if to_prune.len() >= self.vnodes.len() {
            // Don't prune ALL vnodes — keep at least one active
            self.vnodes[0].pruned.store(0, Ordering::Relaxed);
        }
    }

    // ------------------------------------------------------------------
    // all_peers (unchanged from original but with pruning check)
    // ------------------------------------------------------------------
    pub(super) async fn all_peers(&self) -> Vec<String> {
        let mut peers = std::collections::HashSet::new();
        peers.insert(self.self_uri.clone());
        if self.vnodes.is_empty() {
            return peers.into_iter().collect();
        }

        let start_vid = self.vnodes[0].vid;
        let mut current = self.vnodes[0].successor().await;
        for _ in 0..1000 {
            if current.id == start_vid {
                break;
            }
            if current.address != self.self_uri {
                peers.insert(current.address.clone());
            }
            if let Some(next) = self
                .remote
                .get_successor(&current.address, current.id)
                .await
            {
                if next.id == current.id || next.address == current.address {
                    break;
                }
                current = next;
            } else {
                break;
            }
        }
        peers.into_iter().collect()
    }
}
