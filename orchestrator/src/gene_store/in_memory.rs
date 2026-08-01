use crate::gene_store::eviction::EvictionPolicy;
use crate::gene_store::metric::DistanceMetric;
use crate::gene_store::r#trait::GeneStore;
use crate::gene_store::record::GeneRecord;
use tokio::sync::Mutex;

/// Concrete in-process store. Holds a flat `Vec<GeneRecord>` behind a
/// `tokio::sync::Mutex` so the trait stays `&self`-shareable across
/// async tasks.
///
/// Brute-force KNN is O(n·d) per query — fine for typical GA population
/// sizes. When you outgrow it, swap in an HNSW-backed impl behind the
/// same `GeneStore` trait; no caller changes required.
pub struct InMemoryGeneStore<M, E> {
    records: Mutex<Vec<GeneRecord>>,
    metric: M,
    eviction: E,
}

impl<M, E> InMemoryGeneStore<M, E>
where
    M: DistanceMetric,
    E: EvictionPolicy,
{
    pub fn new(metric: M, eviction: E) -> Self {
        Self {
            records: Mutex::new(Vec::new()),
            metric,
            eviction,
        }
    }
}

#[async_trait::async_trait]
impl<M, E> GeneStore for InMemoryGeneStore<M, E>
where
    M: DistanceMetric + 'static,
    E: EvictionPolicy + 'static,
{
    async fn store(&self, genes: Vec<f64>, fitness: f64, generation: usize) {
        let mut records = self.records.lock().await;
        records.push(GeneRecord {
            genes,
            fitness,
            generation,
            last_accessed_gen: generation,
        });
    }

    async fn query_knn(
        &self,
        query: &[f64],
        k: usize,
        current_generation: usize,
    ) -> Vec<GeneRecord> {
        let mut records = self.records.lock().await;

        // Compute distances against all records, then sort ascending.
        let mut scored: Vec<(f64, usize)> = records
            .iter()
            .enumerate()
            .map(|(i, r)| (self.metric.distance(query, &r.genes), i))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(scored.len());

        let mut out = Vec::with_capacity(take);
        for (_, idx) in scored.into_iter().take(take) {
            // Access-refresh: mark retrieved records as seen this generation.
            records[idx].last_accessed_gen = current_generation;
            out.push(records[idx].clone());
        }
        out
    }

    async fn lookup_exact(&self, genes: &[f64], current_generation: usize) -> Option<f64> {
        let mut records = self.records.lock().await;
        for r in records.iter_mut() {
            if r.genes.len() == genes.len() && r.genes.iter().zip(genes.iter()).all(|(a, b)| a == b)
            {
                r.last_accessed_gen = current_generation;
                return Some(r.fitness);
            }
        }
        None
    }

    async fn evict_expired(&self, current_generation: usize) {
        let mut records = self.records.lock().await;
        records.retain(|r| !self.eviction.should_evict(r, current_generation));
    }
}
