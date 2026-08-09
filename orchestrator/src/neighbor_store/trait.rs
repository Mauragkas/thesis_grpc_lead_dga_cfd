use crate::lead_store::GenePayload;
use tonic::Status;

/// ISP: what the GA needs from a persistent nearest-neighbor store backed
/// by the LEAD DHT. OCP/DIP: new strategies (single-curve, LSH, hybrid)
/// implement this without touching GA code.
#[async_trait::async_trait]
pub trait NeighborStore: Send + Sync {
    /// Persist one evaluated individual under its Hilbert-embedded keys.
    async fn store(&self, genes: &[f64], fitness: f64, generation: usize) -> Result<(), Status>;

    /// Return the `k` most-similar stored individuals to `query`,
    /// most-similar first, using multi-probe Hilbert range queries.
    async fn query_knn(&self, query: &[f64], k: usize) -> Result<Vec<GenePayload>, Status>;
}
