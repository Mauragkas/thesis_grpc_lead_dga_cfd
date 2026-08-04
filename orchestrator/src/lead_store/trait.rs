use tonic::Status;

/// Narrow storage contract for persisting genes to a LEAD node.
/// OCP/DIP: a mock or alternative transport can implement this without
/// touching the GA loop.
#[async_trait::async_trait]
pub trait LeadStore: Send + Sync {
    async fn store_gene(&self, key: &str, value: &str) -> Result<(), Status>;
    async fn get_gene(&self, key: &str) -> Result<Option<String>, Status>;
}
