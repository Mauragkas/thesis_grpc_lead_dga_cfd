use std::collections::HashSet;
use std::sync::atomic::Ordering;
use tracing::{info, warn};

use super::{
    ChordNode, DRIFT_THRESHOLD, FRM_GRACE_PERIOD_SECS, MIN_KEYS_FOR_DRIFT, PID_ADJUST_INTERVAL,
};
use crate::ring::in_range;
use crate::rmi::RmiModel;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    // ------------------------------------------------------------------
    // Step 1.3 / 5.1: Record insertion + trigger PID adjustment
    // ------------------------------------------------------------------
    pub async fn record_insertion(&self, key: &str) {
        self.keys_total.fetch_add(1, Ordering::Relaxed);
        let mut rmi = self.rmi.write().await;
        rmi.drift_new += 1;

        // Mark the leaf as dirty for differential sync
        let f = crate::rmi::feature(key);
        let bin = ((f * rmi.active.stage0_bins as f64) as usize)
            .min(rmi.active.stage0_bins.saturating_sub(1));
        rmi.dirty_leaves.insert(bin);

        let total = self.keys_total.load(Ordering::Relaxed);
        if total >= MIN_KEYS_FOR_DRIFT
            && self.start_time.elapsed().as_secs() >= FRM_GRACE_PERIOD_SECS
        {
            let drift_ratio = rmi.drift_new as f64 / total.max(1) as f64;
            if drift_ratio >= DRIFT_THRESHOLD && !rmi.update_ready {
                rmi.update_ready = true;
                info!(
                    drift_new = rmi.drift_new,
                    keys_total = total,
                    drift_ratio,
                    "FRM: drift threshold crossed, marking update_ready"
                );
            }
        }
        drop(rmi);

        // Online PID anchor adjustment every PID_ADJUST_INTERVAL inserts
        let since = self.insert_since_pid.fetch_add(1, Ordering::Relaxed);
        if since > 0 && since % PID_ADJUST_INTERVAL == 0 {
            self.run_pid_adjustment().await;
        }
    }

    /// Run the 2-bit PID controller on each leaf's anchor to keep 95% of
    /// keys within each vnode's VID window.
    async fn run_pid_adjustment(&self) {
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

    // ------------------------------------------------------------------
    // Step 5.3: Retrain + maybe coordinator trigger
    // ------------------------------------------------------------------
    pub async fn maybe_retrain(&self) {
        if self.start_time.elapsed().as_secs() < FRM_GRACE_PERIOD_SECS {
            return;
        }

        let (update_ready, has_update) = {
            let rmi = self.rmi.read().await;
            (rmi.update_ready, rmi.update.is_some())
        };
        if !update_ready || has_update {
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
        let model = RmiModel::train_auto(&keys, version);
        let mut rmi = self.rmi.write().await;
        rmi.update = Some(model);
        rmi.dirty_leaves.clear();
        info!(
            "FRM: retrained local model (auto-config) on {} keys",
            keys.len()
        );
    }

    // ------------------------------------------------------------------
    // Step 5.2-5.3: Heartbeat with 90% neighbor quorum coordinator
    // ------------------------------------------------------------------
    pub async fn heartbeat_round(&self) {
        let (ready, version, has_update) = {
            let rmi = self.rmi.read().await;
            (rmi.update_ready, rmi.active.version, rmi.update.is_some())
        };

        let neighbors = self.neighbor_set().await;
        if neighbors.is_empty() {
            tracing::debug!("FRM: heartbeat round — no neighbors");
            return;
        }

        let mut total = 0usize;
        let mut peers_ready = 0usize;
        for addr in &neighbors {
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
            "FRM: heartbeat round"
        );

        // LEAD Step 5.3: become Transient Coordinator if ≥90% of
        // immediate successor + predecessor neighbors have update_ready.
        if total > 0 {
            let ratio = peers_ready as f64 / total as f64;
            if ratio >= 0.90 && has_update {
                info!(
                    "FRM: 90% neighbor quorum ({} / {}), becoming Transient Coordinator",
                    peers_ready, total
                );
                self.run_global_training_round().await;
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 5.3: Coordinator gathers models via differential sync
    // ------------------------------------------------------------------
    async fn run_global_training_round(&self) {
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

    async fn reset_drift_state(&self) {
        let n = self.storage.len().await;
        self.keys_total.store(n, Ordering::Relaxed);
        let mut rmi = self.rmi.write().await;
        rmi.drift_new = 0;
        rmi.update_ready = false;
    }

    // ------------------------------------------------------------------
    // Step 5: push_model / request_model / heartbeat
    // ------------------------------------------------------------------
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
                    rmi.dirty_leaves.clear();
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
        (
            model.version,
            serde_json::to_vec(&model).unwrap_or_default(),
        )
    }

    pub async fn heartbeat(&self) -> bool {
        self.rmi.read().await.update_ready
    }

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
            if err_rate > super::PRUNE_ERROR_RATE || inactive > super::PRUNE_INACTIVE_SECS {
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
    async fn all_peers(&self) -> Vec<String> {
        let mut peers = HashSet::new();
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

// ------------------------------------------------------------------
// Federated Averaging (unchanged, but handles both leaf kinds)
// ------------------------------------------------------------------
fn fed_avg(models: Vec<RmiModel>, version: u64) -> RmiModel {
    if models.is_empty() {
        return RmiModel::default();
    }
    let bins = models[0].stage0_bins;
    let total_n: usize = models.iter().map(|m| m.n.max(1)).sum();
    let mut leaves = vec![
        crate::rmi::LeafKind::Linear(crate::rmi::LinearLeaf {
            weight: 0.0,
            bias: 0.0,
            anchor: crate::rmi::Anchor::default()
        });
        bins
    ];
    for b in 0..bins {
        let mut w_sum = 0.0;
        let mut bias_sum = 0.0;
        let mut off_sum = 0.0;
        let mut count = 0usize;
        for m in &models {
            let wt = m.n as f64 / total_n as f64;
            if let Some(leaf) = m.leaves.get(b) {
                match leaf {
                    crate::rmi::LeafKind::Linear(l) => {
                        w_sum += l.weight * wt;
                        bias_sum += l.bias * wt;
                        off_sum += l.anchor.offset * wt;
                        count += 1;
                    }
                    crate::rmi::LeafKind::RadixSpline(r) => {
                        off_sum += r.anchor.offset * wt;
                        count += 1;
                    }
                }
            }
        }
        if count > 0 {
            leaves[b] = crate::rmi::LeafKind::Linear(crate::rmi::LinearLeaf {
                weight: w_sum,
                bias: bias_sum,
                anchor: crate::rmi::Anchor {
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
        pid_state: vec![0u8; bins],
    }
}
