//! SRP: migration tunables only. No behaviour.

use std::time::Duration;

/// How often and how much to migrate between orchestrator islands.
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Migrate every N generations.
    pub interval_generations: usize,
    /// Number of individuals to send per migration event.
    pub migrant_count: usize,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            interval_generations: 5,
            migrant_count: 3,
        }
    }
}

/// Ring-level tunables, separate from migration.
#[derive(Debug, Clone)]
pub struct RingConfig {
    /// Address this node binds to for gRPC (e.g. "0.0.0.0:50060").
    pub bind_address: String,
    /// Address other nodes use to reach us (e.g. "orchestrator:50060").
    pub self_address: String,
    /// Bootstrap node to join through, if any.
    pub bootstrap_address: Option<String>,
    /// How often to run stabilization.
    pub stabilize_interval: Duration,
    /// Number of successors to track for fault tolerance.
    pub successor_list_size: usize,
}

impl Default for RingConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:50060".to_string(),
            self_address: "orchestrator:50060".to_string(),
            bootstrap_address: None,
            stabilize_interval: Duration::from_secs(5),
            successor_list_size: 8,
        }
    }
}
