pub mod grpc;
pub mod r#trait;

pub use grpc::GrpcLeadStore;
pub use r#trait::LeadStore;

use serde::{Deserialize, Serialize};
use tracing::warn;

/// Payload persisted into the LEAD DHT for each evaluated individual.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenePayload {
    pub genes: Vec<f64>,
    pub fitness: f64,
    pub generation: usize,
}

/// Deterministic string key for a gene vector. The lead node applies its
/// learned hash to this string to pick the owning virtual node.
pub fn gene_key(genes: &[f64]) -> String {
    serde_json::to_string(genes).unwrap_or_else(|e| {
        warn!("Failed to serialize gene vector for key: {e}; using empty key");
        String::new()
    })
}
