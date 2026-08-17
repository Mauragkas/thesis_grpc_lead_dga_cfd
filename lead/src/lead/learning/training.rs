use tracing::{info, warn};

use crate::rmi::RmiModel;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::LeadNode;
use super::fed_avg::fed_avg;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    // ------------------------------------------------------------------
    // Step 5.3: Coordinator gathers models via differential sync
    // ------------------------------------------------------------------
    pub(super) async fn run_global_training_round(&self) {
        info!("FRM: starting global training round as Transient Coordinator");

        // Own model is always the seed; neighboring models are appended.
        let mut models = vec![self.learning.current_model().await];
        self.collect_neighbor_models(&mut models).await;

        if models.is_empty() {
            warn!("FRM: global round aborted — no models");
            return;
        }

        let global = self.build_global_model(models).await;
        self.broadcast_global_model(&global).await;

        let version = global.version;
        self.learning.activate(global).await;

        info!("FRM: global model v{} activated, migrating keys", version);
        self.learning.set_keys_total(self.storage.len().await);
        self.migrate_keys_for_new_model().await;
    }

    /// Pull the current (or pending) model from each remote neighbor.
    async fn collect_neighbor_models(&self, models: &mut Vec<RmiModel>) {
        let neighbors = self.neighbor_set().await;
        let active_ver = self.learning.version().await;
        for addr in &neighbors {
            if *addr == self.self_uri {
                continue;
            }
            match self.remote.request_model(addr, &self.self_uri).await {
                Some((version, data)) if version >= active_ver => {
                    match serde_json::from_slice::<RmiModel>(&data) {
                        Ok(m) => {
                            tracing::debug!(peer=%addr, version, n=m.n, "FRM: collected");
                            models.push(m);
                        }
                        Err(e) => warn!("FRM: bad model from {addr}: {e}"),
                    }
                }
                Some(_) => {} // stale version, skip
                None => warn!("FRM: request_model from {addr} failed"),
            }
        }
    }

    /// Federate collected models into a single next-version global model.
    async fn build_global_model(&self, models: Vec<RmiModel>) -> RmiModel {
        let max_version = models.iter().map(|m| m.version).max().unwrap_or(1);
        let new_version = max_version + 1;
        fed_avg(models, new_version)
    }

    /// Serialize and push the global model to all known peers.
    async fn broadcast_global_model(&self, global: &RmiModel) {
        let data = serde_json::to_vec(global).unwrap_or_default();
        let peers = self.all_peers().await;
        for addr in &peers {
            if *addr == self.self_uri {
                continue;
            }
            if !self.remote.push_model(addr, global.version, &data).await {
                warn!("FRM: push_model to {addr} rejected");
            }
        }
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
        self.learning.reset_drift(n).await;
    }
}
