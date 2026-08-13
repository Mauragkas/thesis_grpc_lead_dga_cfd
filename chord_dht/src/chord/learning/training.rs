use std::sync::atomic::Ordering;
use tracing::{info, warn};

use crate::rmi::RmiModel;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::ChordNode;
use super::fed_avg::fed_avg;

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    // ------------------------------------------------------------------
    // Step 5.3: Coordinator gathers models via differential sync
    // ------------------------------------------------------------------
    pub(super) async fn run_global_training_round(&self) {
        info!("FRM: starting global training round as Transient Coordinator");

        let neighbors: Vec<String> = self.neighbor_set().await.into_iter().collect();
        let mut models: Vec<RmiModel> = Vec::new();

        // Collect own model
        {
            let rmi = self.rmi.read().await;
            models.push(rmi.update.clone().unwrap_or_else(|| rmi.active.clone()));
        }

        // Request differential (changed leaves only) from neighbors
        for addr in &neighbors {
            if *addr == self.self_uri {
                continue;
            }
            match self.remote.request_model(&addr, &self.self_uri).await {
                Some((version, data)) => {
                    let active_ver = self.rmi.read().await.active.version;
                    if version >= active_ver {
                        match serde_json::from_slice::<RmiModel>(&data) {
                            Ok(m) => {
                                tracing::debug!(peer=%addr, version, n=m.n, "FRM: collected");
                                models.push(m);
                            }
                            Err(e) => warn!("FRM: bad model from {addr}: {e}"),
                        }
                    }
                }
                None => warn!("FRM: request_model from {addr} failed"),
            }
        }

        if models.is_empty() {
            warn!("FRM: global round aborted — no models");
            return;
        }

        let max_version = models.iter().map(|m| m.version).max().unwrap_or(1);
        let new_version = max_version + 1;
        let global = fed_avg(models, new_version);
        let data = serde_json::to_vec(&global).unwrap_or_default();

        // Broadcast to all known peers (not just neighbors)
        let peers = self.all_peers().await;
        for addr in &peers {
            if *addr == self.self_uri {
                continue;
            }
            if !self.remote.push_model(addr, global.version, &data).await {
                warn!("FRM: push_model to {addr} rejected");
            }
        }

        {
            let mut rmi = self.rmi.write().await;
            rmi.active = global;
            rmi.update = None;
            rmi.drift_new = 0;
            rmi.update_ready = false;
            rmi.dirty_leaves.clear();
        }

        info!(
            "FRM: global model v{} activated, migrating keys",
            new_version
        );
        self.keys_total
            .store(self.storage.len().await, Ordering::Relaxed);
        self.migrate_keys_for_new_model().await;
    }

    // ------------------------------------------------------------------
    // Key migration after model update
    // ------------------------------------------------------------------
    pub(super) async fn migrate_keys_for_new_model(&self) {
        let snap = self.storage.snapshot().await;
        let mut migrated = 0usize;
        let mut kept = 0usize;
        for (k, v) in snap {
            if !self.owns_key(&k).await {
                let target = self.lookup_target(&k).await;
                if target.address != self.self_uri {
                    if self.remote.put_local(&target.address, &k, &v).await {
                        self.storage.remove(&k).await;
                        migrated += 1;
                    } else {
                        warn!("FRM: migration put_local failed for key={k}");
                    }
                } else {
                    kept += 1;
                }
            } else {
                kept += 1;
            }
        }
        info!(
            "FRM: migrated {}, kept {} after model update",
            migrated, kept
        );
        self.reset_drift_state().await;
    }

    pub(super) async fn reset_drift_state(&self) {
        let n = self.storage.len().await;
        self.keys_total.store(n, Ordering::Relaxed);
        let mut rmi = self.rmi.write().await;
        rmi.drift_new = 0;
        rmi.update_ready = false;
    }
}
