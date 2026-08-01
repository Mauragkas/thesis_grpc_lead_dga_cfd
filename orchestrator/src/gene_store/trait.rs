use crate::gene_store::record::GeneRecord;

/// ISP: a narrow storage + retrieval contract.
///
/// OCP / DIP: a future SQLite / Postgres / HNSW-backed store implements
/// this trait without modifying GA code. `GaRunner` depends on this
/// abstraction, never on `InMemoryGeneStore`.
#[async_trait::async_trait]
pub trait GeneStore: Send + Sync {
    /// Persist one evaluated individual. `generation` is the GA generation
    /// at which the evaluation occurred.
    async fn store(&self, genes: Vec<f64>, fitness: f64, generation: usize);

    /// Return the `k` most-similar records to `query`, most-similar first.
    /// Touching a record refreshes its `last_accessed_gen` to
    /// `current_generation` (access-refresh TTL semantics).
    async fn query_knn(
        &self,
        query: &[f64],
        k: usize,
        current_generation: usize,
    ) -> Vec<GeneRecord>;

    /// Exact-match lookup. Returns the stored fitness if a record with
    /// bit-identical genes exists, refreshing its `last_accessed_gen`.
    /// Used to short-circuit re-evaluation of individuals seen before
    /// (e.g. elite survivors carried over unchanged).
    async fn lookup_exact(&self, genes: &[f64], current_generation: usize) -> Option<f64>;

    /// Drop records whose retention policy says they have expired.
    async fn evict_expired(&self, current_generation: usize);
}
