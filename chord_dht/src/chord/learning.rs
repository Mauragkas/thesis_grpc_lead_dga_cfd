use std::collections::HashSet;
use std::sync::atomic::Ordering;
use tracing::{info, warn};

use crate::rmi::{LinearLeaf, RmiModel};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::ChordNode;

const DRIFT_THRESHOLD: f64 = 0.40;
const MIN_KEYS_FOR_DRIFT: usize = 50;

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub async fn record_insertion(&self, _key: &str) {
        self.keys_total.fetch_add(1, Ordering::Relaxed);
        let mut rmi = self.rmi.write().await;
        rmi.drift_new += 1;
        let total = self.keys_total.load(Ordering::Relaxed);
        if total >= MIN_KEYS_FOR_DRIFT {
            let drift_ratio = rmi.drift_new as f64 / total.max(1) as f64;
            if drift_ratio >= DRIFT_THRESHOLD && !rmi.update_ready {
                rmi.update_ready = true;
                info!(
                    drift_new = rmi.drift_new,
                    keys_total = total,
                    drift_ratio,
                    "FRM: drift threshold crossed, marking update_ready"
                );
            } else {
                tracing::debug!(
                    drift_new = rmi.drift_new,
                    keys_total = total,
                    drift_ratio,
                    "FRM: insertion recorded"
                );
            }
        }
    }

    pub async fn maybe_retrain(&self) {
        let (update_ready, has_update, drift_new, keys_total, active_ver) = {
            let rmi = self.rmi.read().await;
            (
                rmi.update_ready,
                rmi.update.is_some(),
                rmi.drift_new,
                self.keys_total.load(Ordering::Relaxed),
                rmi.active.version,
            )
        };

        if !update_ready {
            return;
        }

        if has_update {
            tracing::debug!(
                active_ver,
                drift_new,
                keys_total,
                "FRM: update_ready=true but local model already pending, skipping retrain"
            );
            return;
        }

        let keys: Vec<String> = self
            .storage
            .snapshot()
            .await
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        if keys.is_empty() {
            tracing::debug!("FRM: update_ready=true but storage empty, skipping retrain");
            return;
        }

        let version = self.model_version_counter.load(Ordering::SeqCst);
        tracing::debug!(
            active_ver,
            drift_new,
            keys_total,
            keys_in_storage = keys.len(),
            "FRM: drift threshold crossed, training local model"
        );
        let model = RmiModel::train(&keys, version);
        self.rmi.write().await.update = Some(model);
        info!(
            "FRM: retrained local model on {} keys (drift_new={}, keys_total={}, active_ver={})",
            keys.len(),
            drift_new,
            keys_total,
            active_ver
        );
    }

    pub async fn heartbeat_round(&self) {
        let (ready, version, has_update) = {
            let rmi = self.rmi.read().await;
            (rmi.update_ready, rmi.active.version, rmi.update.is_some())
        };

        let peers = self.all_peers().await;
        if peers.is_empty() {
            tracing::debug!("FRM: heartbeat round — no peers");
            return;
        }

        tracing::debug!(
            ready,
            has_update,
            version,
            peer_count = peers.len(),
            "FRM: heartbeat round starting"
        );

        let mut total = 0usize;
        let mut peers_ready = 0usize;
        for addr in &peers {
            if let Some(ok) = self
                .remote
                .heartbeat(addr, &self.self_uri, ready, version)
                .await
            {
                total += 1;
                if ok {
                    peers_ready += 1;
                }
            } else {
                warn!("FRM: heartbeat to {addr} failed");
            }
        }

        tracing::debug!(
            total,
            peers_ready,
            ready,
            has_update,
            "FRM: heartbeat round complete"
        );

        if ready && has_update && total > 0 {
            let max_addr = peers.iter().max().cloned().unwrap_or_default();
            let is_leader = self.self_uri == max_addr;
            tracing::debug!(
                is_leader,
                leader = %max_addr,
                self_uri = %self.self_uri,
                "FRM: leader election check"
            );
            if is_leader {
                self.run_global_training_round().await;
            }
        }
    }

    /// Collect all distinct node addresses from the ring.
    /// Traverses the successor chain starting from the first vnode until
    /// we loop back to the start.
    async fn all_peers(&self) -> Vec<String> {
        let mut peers = HashSet::new();
        peers.insert(self.self_uri.clone());

        if self.vnodes.is_empty() {
            return peers.into_iter().collect();
        }

        let start_vid = self.vnodes[0].vid;
        let mut current = self.vnodes[0].successor().await;

        // Walk the ring until we come back to start_vid
        for _ in 0..1000 {
            // safety limit
            if current.id == start_vid {
                break;
            }
            if current.address != self.self_uri {
                peers.insert(current.address.clone());
            }
            // Ask the current node for its successor of the same vnode
            if let Some(next) = self
                .remote
                .get_successor(&current.address, current.id)
                .await
            {
                if next.id == current.id || next.address == current.address {
                    break; // alone or stuck
                }
                current = next;
            } else {
                break;
            }
        }
        peers.into_iter().collect()
    }

    /// Leader-driven global training round:
    /// 1. Collect models from ALL peers (not just neighbors).
    /// 2. Compute federated average.
    /// 3. Push the global model to all peers.
    /// 4. Activate locally and migrate keys.
    async fn run_global_training_round(&self) {
        info!("FRM: starting global training round as leader");

        let peers = self.all_peers().await;
        let mut models: Vec<RmiModel> = Vec::new();

        {
            let rmi = self.rmi.read().await;
            models.push(rmi.update.clone().unwrap_or_else(|| rmi.active.clone()));
        }
        tracing::debug!(own_model = true, "FRM: collected own model");

        for addr in &peers {
            if *addr == self.self_uri {
                continue;
            }
            match self.remote.request_model(addr, &self.self_uri).await {
                Some((version, data)) => {
                    let active_ver = self.rmi.read().await.active.version;
                    if version >= active_ver {
                        match serde_json::from_slice::<RmiModel>(&data) {
                            Ok(m) => {
                                tracing::debug!(
                                    peer = %addr,
                                    version,
                                    n = m.n,
                                    "FRM: collected peer model"
                                );
                                models.push(m);
                            }
                            Err(e) => warn!("FRM: bad model from {addr}: {e}"),
                        }
                    } else {
                        tracing::debug!(
                            peer = %addr,
                            version,
                            active_ver,
                            "FRM: peer model stale, skipping"
                        );
                    }
                }
                None => warn!("FRM: request_model from {addr} failed"),
            }
        }

        if models.is_empty() {
            warn!("FRM: global round aborted — no models collected");
            return;
        }

        let max_version = models.iter().map(|m| m.version).max().unwrap_or(1);
        let new_version = max_version + 1;
        let total_keys: usize = models.iter().map(|m| m.n).sum();
        tracing::debug!(
            models_collected = models.len(),
            max_version,
            new_version,
            total_keys,
            "FRM: computing federated average"
        );
        let global_model = fed_avg(models, new_version);
        let data = serde_json::to_vec(&global_model).unwrap_or_default();

        for addr in &peers {
            if *addr == self.self_uri {
                continue;
            }
            match self
                .remote
                .push_model(addr, global_model.version, &data)
                .await
            {
                true => {
                    tracing::debug!(peer = %addr, version = global_model.version, "FRM: pushed model")
                }
                false => warn!("FRM: push_model to {addr} rejected/failed"),
            }
        }

        {
            let mut rmi = self.rmi.write().await;
            rmi.active = global_model;
            rmi.update = None;
            rmi.drift_new = 0;
            rmi.update_ready = false;
        }

        info!(
            "FRM: global model version {} activated, migrating keys",
            new_version
        );
        self.migrate_keys_for_new_model().await;
    }

    // After a model update, scan local storage and forward keys that are
    // no longer owned by this node to their new owner.
    async fn migrate_keys_for_new_model(&self) {
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
                        warn!(
                            "FRM: migration put_local failed for key={k} -> {}",
                            target.address
                        );
                    }
                } else {
                    kept += 1;
                }
            } else {
                kept += 1;
            }
        }
        info!(
            "FRM: migrated {} keys after model update (kept {})",
            migrated, kept
        );
        self.reset_drift_state().await;
    }

    /// Reset drift counters to reflect the current, settled storage contents.
    /// This prevents migration-induced inserts from falsely triggering retraining.
    async fn reset_drift_state(&self) {
        let n = self.storage.len().await;
        self.keys_total.store(n, Ordering::Relaxed);
        let mut rmi = self.rmi.write().await;
        rmi.drift_new = 0;
        rmi.update_ready = false;
        tracing::debug!(keys_total = n, "FRM: drift state reset after migration");
    }

    pub async fn push_model(&self, version: u64, data: &[u8]) -> bool {
        let accepted = {
            let mut rmi = self.rmi.write().await;
            if version <= rmi.active.version {
                warn!(
                    "reject model rollback have={} got={}",
                    rmi.active.version, version
                );
                return false;
            }
            match serde_json::from_slice::<RmiModel>(data) {
                Ok(m) => {
                    rmi.active = m;
                    rmi.update = None;
                    rmi.drift_new = 0;
                    rmi.update_ready = false;
                    true
                }
                Err(e) => {
                    warn!("FRM: bad model payload: {e}");
                    false
                }
            }
        };
        if accepted {
            info!("FRM: accepted model version {}", version);
            self.migrate_keys_for_new_model().await;
        }
        accepted
    }

    pub async fn request_model(&self) -> (u64, Vec<u8>) {
        let rmi = self.rmi.read().await;
        let model = rmi.update.clone().unwrap_or_else(|| rmi.active.clone());
        let version = model.version;
        (version, serde_json::to_vec(&model).unwrap_or_default())
    }

    pub async fn heartbeat(&self) -> bool {
        self.rmi.read().await.update_ready
    }
}

/// Federated Averaging over leaf parameters, weighted by training-set size.
fn fed_avg(models: Vec<RmiModel>, version: u64) -> RmiModel {
    if models.is_empty() {
        return RmiModel::default();
    }
    let bins = models[0].stage0_bins;
    let total_n: usize = models.iter().map(|m| m.n.max(1)).sum();
    let mut leaves = vec![
        LinearLeaf {
            weight: 0.0,
            bias: 0.0,
            anchor: crate::rmi::Anchor::default()
        };
        bins
    ];
    for b in 0..bins {
        let mut w = 0.0;
        let mut bias = 0.0;
        let mut offset = 0.0;
        for m in &models {
            let wt = m.n as f64 / total_n as f64;
            let lf = &m.leaves[b];
            w += lf.weight * wt;
            bias += lf.bias * wt;
            offset += lf.anchor.offset * wt;
        }
        leaves[b] = LinearLeaf {
            weight: w,
            bias,
            anchor: crate::rmi::Anchor { offset, scale: 1.0 },
        };
    }
    let n: usize = models.iter().map(|m| m.n).sum();
    RmiModel {
        stage0_bins: bins,
        leaves,
        n,
        version,
    }
}
