//! Per-individual retry policy with exponential backoff and jitter.
//!
//! Provides intelligent retry scheduling and transient error classification
//! for individual evaluation requests.

use rand::Rng;
use std::time::Duration;
use tonic::{Code, Status};

/// Configuration and calculation for exponential backoff with jitter.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub jitter_factor: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(3000),
            jitter_factor: 0.25,
        }
    }
}

impl RetryPolicy {
    pub fn new(
        max_attempts: usize,
        initial_backoff: Duration,
        max_backoff: Duration,
        jitter_factor: f64,
    ) -> Self {
        Self {
            max_attempts,
            initial_backoff,
            max_backoff,
            jitter_factor: jitter_factor.clamp(0.0, 1.0),
        }
    }

    /// Computes exponential backoff with randomized jitter for a given attempt (1-indexed).
    pub fn backoff_for_attempt(&self, attempt: usize) -> Duration {
        if attempt <= 1 {
            return self.apply_jitter(self.initial_backoff);
        }

        let factor = 2u64.saturating_pow((attempt - 1).min(10) as u32);
        let exp_millis = self
            .initial_backoff
            .as_millis()
            .saturating_mul(factor as u128);
        let capped_millis = exp_millis.min(self.max_backoff.as_millis());
        self.apply_jitter(Duration::from_millis(capped_millis as u64))
    }

    fn apply_jitter(&self, base: Duration) -> Duration {
        if self.jitter_factor <= 0.0 {
            return base;
        }
        let base_f = base.as_secs_f64();
        let mut rng = rand::thread_rng();
        // Jitter between [1 - jitter, 1 + jitter]
        let delta = (rng.gen::<f64>() * 2.0 - 1.0) * self.jitter_factor;
        let jittered_f = (base_f * (1.0 + delta)).max(0.001);
        Duration::from_secs_f64(jittered_f)
    }

    /// Determines if a gRPC Status code represents a transient, retryable failure.
    pub fn is_retryable(status: &Status) -> bool {
        matches!(
            status.code(),
            Code::Unavailable
                | Code::DeadlineExceeded
                | Code::ResourceExhausted
                | Code::Aborted
                | Code::Unknown
                | Code::Cancelled
        )
    }
}
