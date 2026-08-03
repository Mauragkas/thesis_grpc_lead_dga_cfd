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
            if drift_ratio >= DRIFT_THRESHOLD {
                rmi.update_ready = true;
            }
        }
    }

    pub async fn maybe_retrain(&self) {
        if !self.rmi.read().await.update_ready {
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
            return;
        }
        let version = self.model_version_counter.load(Ordering::SeqCst);
        let model = RmiModel::train(&keys, version);
        self.rmi.write().await.update = Some(model);
        info!("FRM: retrained update_rmi on {} keys", keys.len());
    }

    pub async fn heartbeat_round(&self) {
        let (ready, version, has_update) = {
            let rmi = self.rmi.read().await;
            (rmi.update_ready, rmi.active.version, rmi.update.is_some())
        };
        let neighbors = self.immediate_neighbors().await;
        if neighbors.is_empty() {
            return;
        }
        let mut total = 0usize;
        for addr in &neighbors {
            if self
                .remote
                .heartbeat(addr, &self.self_uri, ready, version)
                .await
                .is_some()
            {
                total += 1;
            }
        }
        if ready && has_update && total > 0 {
            self.assume_coordinator().await;
        }
    }

    async fn immediate_neighbors(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        for vnode in &self.vnodes {
            let succ = vnode.successor().await;
            if succ.id != vnode.vid && succ.address != self.self_uri {
                set.insert(succ.address.clone());
            }
            if let Some(p) = vnode.predecessor.read().await.clone() {
                if p.id != vnode.vid && p.address != self.self_uri {
                    set.insert(p.address.clone());
                }
            }
        }
        set
    }

    async fn assume_coordinator(&self) {
        info!("FRM: assuming transient coordinator role");
        let neighbors = self.immediate_neighbors().await;

        let mut params: Vec<RmiModel> = Vec::new();
        {
            let rmi = self.rmi.read().await;
            params.push(rmi.update.clone().unwrap_or_else(|| rmi.active.clone()));
        }
        for addr in &neighbors {
            if let Some((version, data)) = self.remote.request_model(addr, &self.self_uri).await {
                let active_ver = self.rmi.read().await.active.version;
                if version >= active_ver {
                    if let Ok(m) = serde_json::from_slice::<RmiModel>(&data) {
                        params.push(m);
                    }
                }
            }
        }
        if params.is_empty() {
            return;
        }
        let new_version = self.model_version_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let new_model = fed_avg(params, new_version);
        let data = serde_json::to_vec(&new_model).unwrap_or_default();
        for addr in &neighbors {
            let _ = self.remote.push_model(addr, new_model.version, &data).await;
        }
        let model_version = new_model.version;
        {
            let mut rmi = self.rmi.write().await;
            rmi.active = new_model;
            rmi.update = None;
            rmi.drift_new = 0;
            rmi.update_ready = false;
        }
        info!("FRM: broadcast model version {}", model_version);
        self.migrate_keys_for_new_model().await;
    }

    /// After a model update, scan local storage and forward keys that are
    /// no longer owned by this node to their new owner.
    async fn migrate_keys_for_new_model(&self) {
        let snap = self.storage.snapshot().await;
        let mut migrated = 0usize;
        for (k, v) in snap {
            if !self.owns_key(&k).await {
                let target = self.lookup_target(&k).await;
                if target.address != self.self_uri {
                    if self.remote.put_local(&target.address, &k, &v).await {
                        self.storage.remove(&k).await;
                        migrated += 1;
                    }
                }
            }
        }
        if migrated > 0 {
            info!("FRM: migrated {} keys after model update", migrated);
        }
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
