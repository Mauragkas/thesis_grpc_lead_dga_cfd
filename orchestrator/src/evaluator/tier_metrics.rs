use std::sync::atomic::{AtomicUsize, Ordering};

/// Single Responsibility: thread-safe telemetry and metrics tracker for
/// Multi-Tier (ε-Bypass) evaluation pipeline.
#[derive(Debug, Default)]
pub struct TierMetricsTracker {
    tier1_exact_hits: AtomicUsize,
    tier2_surrogate_hits: AtomicUsize,
    tier3_simulator_evals: AtomicUsize,
    total_evaluations: AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
pub struct TierMetricsSnapshot {
    pub tier1_exact_hits: usize,
    pub tier2_surrogate_hits: usize,
    pub tier3_simulator_evals: usize,
    pub total_evaluations: usize,
}

impl TierMetricsSnapshot {
    pub fn tier1_ratio(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.tier1_exact_hits as f64 / self.total_evaluations as f64
        }
    }

    pub fn tier2_ratio(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.tier2_surrogate_hits as f64 / self.total_evaluations as f64
        }
    }

    pub fn tier3_ratio(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.tier3_simulator_evals as f64 / self.total_evaluations as f64
        }
    }

    /// Theoretical simulation bypass ratio: (Tier 1 + Tier 2) / Total
    pub fn bypass_ratio(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            (self.tier1_exact_hits + self.tier2_surrogate_hits) as f64 / self.total_evaluations as f64
        }
    }
}

impl TierMetricsTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_tier1_hit(&self, count: usize) {
        self.tier1_exact_hits.fetch_add(count, Ordering::Relaxed);
        self.total_evaluations.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_tier2_hit(&self, count: usize) {
        self.tier2_surrogate_hits.fetch_add(count, Ordering::Relaxed);
        self.total_evaluations.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_tier3_eval(&self, count: usize) {
        self.tier3_simulator_evals.fetch_add(count, Ordering::Relaxed);
        self.total_evaluations.fetch_add(count, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> TierMetricsSnapshot {
        TierMetricsSnapshot {
            tier1_exact_hits: self.tier1_exact_hits.load(Ordering::Relaxed),
            tier2_surrogate_hits: self.tier2_surrogate_hits.load(Ordering::Relaxed),
            tier3_simulator_evals: self.tier3_simulator_evals.load(Ordering::Relaxed),
            total_evaluations: self.total_evaluations.load(Ordering::Relaxed),
        }
    }

    pub fn reset_generation(&self) -> TierMetricsSnapshot {
        let snap = self.snapshot();
        self.tier1_exact_hits.store(0, Ordering::Relaxed);
        self.tier2_surrogate_hits.store(0, Ordering::Relaxed);
        self.tier3_simulator_evals.store(0, Ordering::Relaxed);
        self.total_evaluations.store(0, Ordering::Relaxed);
        snap
    }
}
