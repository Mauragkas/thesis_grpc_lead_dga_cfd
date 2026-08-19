use crate::rmi::{Anchor, PidState};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::LeadNode;

/// Configurable 2-bit PID controller for nudging a leaf's anchor so that the
/// target fraction of keys stays inside its vnode's window.
pub struct PidTuner {
    /// Desired fraction of keys inside the window.
    pub target_ratio: f64,
    /// Discrete scale step applied on large corrections.
    pub scale_step: f64,
    /// Offset-centering step applied every adjustment.
    pub centering_step: f64,
    /// Error above which the proportional state saturates high (3).
    pub upper_threshold: f64,
    /// Error above which the proportional state moves to 2.
    pub mid_threshold: f64,
    /// Minimum total samples before the controller acts.
    pub min_samples: usize,
}

impl Default for PidTuner {
    fn default() -> Self {
        Self {
            target_ratio: 0.95,
            scale_step: 0.05,
            centering_step: 0.01,
            upper_threshold: 0.05,
            mid_threshold: 0.02,
            min_samples: 20,
        }
    }
}

impl PidTuner {
    /// Adjust `anchor` and evolve `state` for one leaf based on key counts.
    pub fn adjust(
        &self,
        state: &mut PidState,
        anchor: &mut Anchor,
        keys_in_window: usize,
        keys_outside: usize,
    ) {
        let total = keys_in_window.saturating_add(keys_outside);
        if total < self.min_samples {
            return;
        }

        let actual = keys_in_window as f64 / total.max(1) as f64;
        let error = actual - self.target_ratio;

        let prev_p = state.proportional;
        let i_acc = state.integral;
        let prev_d_sign = state.derivative_sign;
        let prev_d_mag = state.derivative_mag;

        let new_p = if error > self.upper_threshold {
            3u8
        } else if error > self.mid_threshold {
            2u8
        } else if error > -self.mid_threshold {
            1u8
        } else {
            0u8
        };

        // 2-bit PID output = clamp(p_state + i_acc/4 + d)
        let i_contrib = i_acc as f64 / 4.0;
        let d_contrib = if prev_d_sign {
            prev_d_mag as f64
        } else {
            -(prev_d_mag as f64)
        };
        let pid = (new_p as f64) + i_contrib + d_contrib;

        if pid > 2.0 {
            // Large correction: step scale, reset integral accumulator.
            anchor.scale *= 1.0 + self.scale_step;
            state.proportional = new_p;
            state.integral = 1;
            state.derivative_sign = false;
            state.derivative_mag = 0;
        } else if pid < 0.5 {
            anchor.scale *= 1.0 - self.scale_step;
            state.proportional = new_p;
            state.integral = 1;
            state.derivative_sign = false;
            state.derivative_mag = 0;
        } else {
            // Within band: accumulate integral + derivative.
            let new_i = (i_acc.saturating_add(1)).min(15);
            let d = if new_p > prev_p {
                1u8
            } else if new_p < prev_p {
                2u8
            } else {
                0u8
            };
            state.proportional = new_p;
            state.integral = new_i;
            state.derivative_sign = (d & 0b10) != 0;
            state.derivative_mag = d & 0b01;
        }

        // Offset centering: shift to keep the median inside the window.
        if error > 0.0 {
            anchor.offset -= self.centering_step * error.abs().min(1.0);
        } else {
            anchor.offset += self.centering_step * error.abs().min(1.0);
        }
    }
}

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    /// Run the 2-bit PID controller on each leaf's anchor to keep the target
    /// fraction of keys within each vnode's VID window.
    pub(super) async fn run_pid_adjustment(&self) {
        let snap = self.storage.snapshot().await;
        if snap.is_empty() {
            return;
        }

        let model = self.learning.active_model().await;
        let bins = model.stage0_bins;
        let mut in_window = vec![0usize; bins];
        let mut outside = vec![0usize; bins];

        for (k, _) in &snap {
            let f = crate::rmi::feature(k);
            let bin = ((f * bins as f64) as usize).min(bins.saturating_sub(1));
            let hash = model.predict(k);
            if self.owning_vnode_for(hash).await.is_some() {
                in_window[bin] += 1;
            } else {
                outside[bin] += 1;
            }
        }

        let tuner = self.config.pid_tuner();
        self.learning
            .adjust_pid_anchors(&tuner, in_window, outside)
            .await;
    }
}

