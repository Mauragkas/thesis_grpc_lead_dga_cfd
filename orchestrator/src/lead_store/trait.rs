use tonic::Status;

/// ISP: narrow storage contract for persisting genes to a LEAD node.
/// OCP/DIP: a mock or alternative transport can implement this without
/// touching the GA loop.
#[async_trait::async_trait]
pub trait LeadStore: Send + Sync {
    async fn store_gene(&self, key: &str, value: &str) -> Result<(), Status>;
    async fn get_gene(&self, key: &str) -> Result<Option<String>, Status>;

    /// Order-preserving range query: returns up to `count` entries whose
    /// keys sort at/after `start_key`. With Hilbert-embedded keys this
    /// approximates Euclidean nearest-neighbor search.
    async fn range_query(
        &self,
        start_key: &str,
        count: u64,
    ) -> Result<Vec<(String, String)>, Status>;
}
