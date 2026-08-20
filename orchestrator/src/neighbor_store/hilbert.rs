use crate::gene_store::metric::DistanceMetric;
use crate::gene_store::EuclideanDistance;
use crate::hilbert::GeneKeyGenerator;
use crate::lead_store::{GenePayload, LeadStore};
use crate::neighbor_store::NeighborStore;
use std::collections::HashMap;
use tonic::Status;
use tracing::debug;

/// SRP: multi-probe Hilbert nearest-neighbor access to the LEAD DHT.
/// Depends on abstractions only (DIP): `GeneKeyGenerator` for embedding
/// and `LeadStore` for transport.
pub struct HilbertNeighborStore<G, L> {
    keygen: G,
    store: L,
    metric: EuclideanDistance,
}

impl<G, L> HilbertNeighborStore<G, L>
where
    G: GeneKeyGenerator,
    L: LeadStore,
{
    pub fn new(keygen: G, store: L) -> Self {
        Self {
            keygen,
            store,
            metric: EuclideanDistance,
        }
    }
}

#[async_trait::async_trait]
impl<G, L> NeighborStore for HilbertNeighborStore<G, L>
where
    G: GeneKeyGenerator + Send + Sync,
    L: LeadStore + Send + Sync,
{
    async fn store(&self, genes: &[f64], fitness: f64, generation: usize) -> Result<(), Status> {
        let payload = GenePayload {
            genes: genes.to_vec(),
            fitness,
            generation,
        };
        let value = serde_json::to_string(&payload)
            .map_err(|e| Status::internal(format!("serialize payload: {e}")))?;

        // Multi-probe indexing: store under every curve concurrently so any curve's
        // range query can find this individual.
        let keys = self.keygen.keys_for(genes);
        let store_futs = keys.iter().map(|key| self.store.store_gene(key, &value));
        futures::future::try_join_all(store_futs).await?;
        debug!("Stored gene under {} Hilbert keys", keys.len());
        Ok(())
    }

    async fn query_knn(&self, query: &[f64], k: usize) -> Result<Vec<GenePayload>, Status> {
        // Multi-probe: fan out to all curves concurrently, merge + dedupe candidates.
        let mut candidates: HashMap<String, GenePayload> = HashMap::new();
        let keys = self.keygen.keys_for(query);
        let query_futs = keys.iter().map(|key| self.store.range_query(key, k as u64));
        let results = futures::future::join_all(query_futs).await;

        for res in results {
            if let Ok(entries) = res {
                for (k, v) in entries {
                    if let Ok(payload) = serde_json::from_str::<GenePayload>(&v) {
                        candidates.entry(k).or_insert(payload);
                    }
                }
            }
        }

        // Rank by true Euclidean distance (the ground truth the Hilbert
        // embedding approximates).
        let mut scored: Vec<(f64, GenePayload)> = candidates
            .into_values()
            .map(|p| (self.metric.distance(query, &p.genes), p))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(k).map(|(_, p)| p).collect())
    }
}
