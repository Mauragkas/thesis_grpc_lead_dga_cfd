mod fed_avg;
mod pid;
mod prune;
mod training;

use std::sync::atomic::Ordering;
use tracing::{info, warn};

use super::{
    ChordNode, DRIFT_THRESHOLD, FRM_GRACE_PERIOD_SECS, MIN_KEYS_FOR_DRIFT, PID_ADJUST_INTERVAL,
};
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
        if since > 0 && since.is_multiple_of(PID_ADJUST_INTERVAL) {
            self.run_pid_adjustment().await;
        }
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
}
