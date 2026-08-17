use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::lead::vnode::RmiState;
use crate::ring::NodeId;
use crate::rmi::{feature, PidState, RmiModel};

/// Owns all online-learning state for a physical node: the learned model
/// (active + pending), the drift/version counters, and the drift bookkeeping.
///
/// Capsule for the LEARNING concern of [`LeadNode`](crate::lead::LeadNode).
/// All mutation of the model state flows through here so the drift/version/
/// dirty-leaf invariants are enforced in one place. Network I/O, storage
/// access and routing decisions stay on `LeadNode`; this type only tracks and
/// transitions the model state.
pub struct LearnedIndex {
    pub(crate) rmi: RwLock<RmiState>,
    pub(crate) keys_total: AtomicUsize,
    pub(crate) model_version_counter: AtomicU64,
    pub(crate) insert_since_pid: AtomicUsize,
    pub(crate) start_time: Instant,
}

impl LearnedIndex {
    pub(crate) fn new() -> Self {
        Self {
            rmi: RwLock::new(RmiState {
                active: RmiModel::default(),
                update: None,
                drift_new: 0,
                update_ready: false,
                dirty_leaves: HashSet::new(),
            }),
            keys_total: AtomicUsize::new(0),
            model_version_counter: AtomicU64::new(1),
            insert_since_pid: AtomicUsize::new(0),
            start_time: Instant::now(),
        }
    }

    // ------------------------------------------------------------------
    // Reads
    // ------------------------------------------------------------------

    /// Predict the hash for `key` using the currently active model.
    pub(crate) async fn predict(&self, key: &str) -> NodeId {
        self.rmi.read().await.active.predict(key)
    }

    /// Clone of the currently active model.
    pub(crate) async fn active_model(&self) -> RmiModel {
        self.rmi.read().await.active.clone()
    }

    /// The model to use as "current": a pending update if present, else active.
    pub(crate) async fn current_model(&self) -> RmiModel {
        let rmi = self.rmi.read().await;
        rmi.update.clone().unwrap_or_else(|| rmi.active.clone())
    }

    /// Resolve the model for a specific version, falling back to active.
    pub(crate) async fn model_for_version(&self, version: u64) -> RmiModel {
        let rmi = self.rmi.read().await;
        if rmi.active.version == version {
            rmi.active.clone()
        } else if let Some(ref update) = rmi.update {
            if update.version == version {
                update.clone()
            } else {
                rmi.active.clone()
            }
        } else {
            rmi.active.clone()
        }
    }

    pub(crate) async fn version(&self) -> u64 {
        self.rmi.read().await.active.version
    }

    pub(crate) async fn is_update_ready(&self) -> bool {
        self.rmi.read().await.update_ready
    }

    /// `(update_ready, has_pending_update)` snapshot.
    pub(crate) async fn status(&self) -> (bool, bool) {
        let rmi = self.rmi.read().await;
        (rmi.update_ready, rmi.update.is_some())
    }

    pub(crate) async fn dirty_leaf_indices(&self) -> Vec<usize> {
        self.rmi.read().await.dirty_leaves.iter().copied().collect()
    }

    /// True while the startup grace period (before drift detection starts) is
    /// still elapsing.
    pub(crate) fn in_grace(&self, grace_secs: u64) -> bool {
        self.start_time.elapsed().as_secs() < grace_secs
    }

    pub(crate) fn next_version(&self) -> u64 {
        self.model_version_counter.load(Ordering::SeqCst)
    }

    // ------------------------------------------------------------------
    // Mutations
    // ------------------------------------------------------------------

    /// Account for one inserted key: bump counters, mark the affected leaf
    /// dirty, and raise `update_ready` when the drift threshold is crossed.
    ///
    /// Returns `true` when a PID adjustment is due (interval reached).
    pub(crate) async fn record_insert(
        &self,
        key: &str,
        grace_secs: u64,
        min_keys_for_drift: usize,
        drift_threshold: f64,
    ) -> bool {
        self.keys_total.fetch_add(1, Ordering::Relaxed);
        let mut rmi = self.rmi.write().await;
        rmi.drift_new += 1;

        let f = feature(key);
        let bin = ((f * rmi.active.stage0_bins as f64) as usize)
            .min(rmi.active.stage0_bins.saturating_sub(1));
        rmi.dirty_leaves.insert(bin);

        let total = self.keys_total.load(Ordering::Relaxed);
        if total >= min_keys_for_drift && self.start_time.elapsed().as_secs() >= grace_secs {
            let drift_ratio = rmi.drift_new as f64 / total.max(1) as f64;
            if drift_ratio >= drift_threshold && !rmi.update_ready {
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

        let since = self.insert_since_pid.fetch_add(1, Ordering::Relaxed);
        since > 0 && since.is_multiple_of(super::super::PID_ADJUST_INTERVAL)
    }

    /// Store a freshly retrained local model as the pending update.
    pub(crate) async fn set_pending_model(&self, model: RmiModel) {
        let mut rmi = self.rmi.write().await;
        rmi.update = Some(model);
        rmi.dirty_leaves.clear();
    }

    /// Promote `model` to active, clearing all pending/drift state.
    pub(crate) async fn activate(&self, model: RmiModel) {
        let mut rmi = self.rmi.write().await;
        rmi.active = model;
        rmi.update = None;
        rmi.drift_new = 0;
        rmi.update_ready = false;
        rmi.dirty_leaves.clear();
    }

    /// Accept an incoming pushed model after a version + payload check.
    /// Returns `false` (without mutating) on rollback or malformed payload.
    pub(crate) async fn accept_pushed_model(&self, version: u64, data: &[u8]) -> bool {
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
    }

    /// Reset drift counters after a migration / model activation.
    pub(crate) async fn reset_drift(&self, keys_total: usize) {
        self.keys_total.store(keys_total, Ordering::Relaxed);
        let mut rmi = self.rmi.write().await;
        rmi.drift_new = 0;
        rmi.update_ready = false;
    }

    pub(crate) fn set_keys_total(&self, n: usize) {
        self.keys_total.store(n, Ordering::Relaxed);
    }

    /// Run the 2-bit PID tuner over every leaf using per-bin key counts.
    pub(crate) async fn adjust_pid_anchors(&self, in_window: Vec<usize>, outside: Vec<usize>) {
        let mut rmi = self.rmi.write().await;
        let model = &mut rmi.active;
        if model.pid_state.len() < model.stage0_bins {
            model.pid_state.resize(model.stage0_bins, PidState::default());
        }
        let tuner = super::pid::PidTuner::default();
        for b in 0..model.stage0_bins {
            tuner.adjust(
                &mut model.pid_state[b],
                model.leaves[b].anchor_mut(),
                in_window.get(b).copied().unwrap_or(0),
                outside.get(b).copied().unwrap_or(0),
            );
        }
        tracing::debug!("PID adjustment complete across {} bins", model.stage0_bins);
    }
}
