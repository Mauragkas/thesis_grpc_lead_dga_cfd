use tracing::{info, warn};

use crate::lead::LeadNode;
use crate::rmi::RmiModel;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

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

        info!(
            "FRM: coordinator collected {} models (including own) for global round",
            models.len()
        );

        if models.is_empty() {
            warn!("FRM: global round aborted — no models");
            return;
        }

        let global = self.build_global_model(models).await;
        self.broadcast_global_model(&global).await;

        let version = global.version;
        if let Ok(json_str) = serde_json::to_string(&global) {
            self.storage
                .put_meta("active_model".to_string(), json_str)
                .await;
        }
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
                Some((version, data)) => {
                    match serde_json::from_slice::<RmiModel>(&data) {
                        Ok(m) => {
                            // Accept if version >= active_ver OR if the peer holds keys (n > 0),
                            // ensuring rebooted nodes with keys are included in federated training.
                            if version >= active_ver || m.n > 0 {
                                info!(
                                    peer = %addr,
                                    version,
                                    n = m.n,
                                    "FRM: collected model from neighbor"
                                );
                                models.push(m);
                            } else {
                                warn!(
                                    "FRM: neighbor {addr} returned un-trained stale model version {version} < active {active_ver}, skipping"
                                );
                            }
                        }
                        Err(e) => warn!("FRM: bad model from {addr}: {e}"),
                    }
                }
                None => warn!("FRM: request_model from {addr} failed"),
            }
        }
    }

    /// Federate collected models into a single next-version global model.
    async fn build_global_model(&self, models: Vec<RmiModel>) -> RmiModel {
        let active_ver = self.learning.version().await;
        let max_version = models.iter().map(|m| m.version).max().unwrap_or(active_ver);
        let new_version = max_version.max(active_ver) + 1;
        fed_avg(models, new_version)
    }

    /// Serialize and push the global model to all known peers.
    async fn broadcast_global_model(&self, global: &RmiModel) {
        let data = serde_json::to_vec(global).unwrap_or_default();
        let peers = self.neighbor_set().await;
        for addr in &peers {
            if *addr == self.self_uri {
                continue;
            }
            if !self.remote.push_model(addr, global.version, &data).await {
                warn!("FRM: push_model to {addr} rejected");
            }
        }
    }

    /// When a new model is activated, keys whose predicted owner is no
    /// longer us are migrated to their new home via /kv/local/:key.
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
        let len = self.storage.len().await;
        self.learning.reset_drift(len).await;
    }
}
