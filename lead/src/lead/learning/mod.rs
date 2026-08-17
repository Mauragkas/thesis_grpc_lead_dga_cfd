mod fed_avg;
mod index;
mod pid;
mod prune;
mod training;

use tracing::{info, warn};

use super::{
    LeadNode, DRIFT_THRESHOLD, FRM_GRACE_PERIOD_SECS, MIN_KEYS_FOR_DRIFT,
};
use crate::rmi::RmiModel;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

pub use index::LearnedIndex;
pub use pid::PidTuner;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    // ------------------------------------------------------------------
    // Step 1.3 / 5.1: Record insertion + trigger PID adjustment
    // ------------------------------------------------------------------
    pub async fn record_insertion(&self, key: &str) {
        let should_pid = self
            .learning
            .record_insert(key, FRM_GRACE_PERIOD_SECS, MIN_KEYS_FOR_DRIFT, DRIFT_THRESHOLD)
            .await;
        if should_pid {
            self.run_pid_adjustment().await;
        }
    }

    // ------------------------------------------------------------------
    // Step 5.3: Retrain + maybe coordinator trigger
    // ------------------------------------------------------------------
    pub async fn maybe_retrain(&self) {
        if self.learning.in_grace(FRM_GRACE_PERIOD_SECS) {
            return;
        }

        let (update_ready, has_update) = self.learning.status().await;
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

        let version = self.learning.next_version();
        let model = RmiModel::train_auto(&keys, version);
        self.learning.set_pending_model(model).await;
        info!(
            "FRM: retrained local model (auto-config) on {} keys",
            keys.len()
        );
    }

    // ------------------------------------------------------------------
    // Step 5.2-5.3: Heartbeat with 90% neighbor quorum coordinator
    // ------------------------------------------------------------------
    pub async fn heartbeat_round(&self) {
        let (ready, has_update) = self.learning.status().await;
        let version = self.learning.version().await;

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
        let accepted = self.learning.accept_pushed_model(version, data).await;
        if accepted {
            info!("FRM: accepted model version {}", version);
            self.migrate_keys_for_new_model().await;
        }
        accepted
    }

    pub async fn request_model(&self) -> (u64, Vec<u8>) {
        let model = self.learning.current_model().await;
        (
            model.version,
            serde_json::to_vec(&model).unwrap_or_default(),
        )
    }

    pub async fn heartbeat(&self) -> bool {
        self.learning.is_update_ready().await
    }
}
