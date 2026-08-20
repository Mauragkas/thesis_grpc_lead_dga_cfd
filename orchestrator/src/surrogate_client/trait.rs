use tonic::Status;

/// Interface Segregation / Dependency Inversion: abstract contract for
/// communicating with the external surrogate model service.
/// High-level evaluation logic depends on this trait, not on concrete gRPC types.
#[async_trait::async_trait]
pub trait SurrogateClient: Send + Sync {
    /// Predicts fitness values for a batch of candidate individuals.
    /// Returns `Ok(Some(fitnesses))` if the surrogate model is ready and produced predictions,
    /// or `Ok(None)` if the surrogate is still in cold-start / uninitialized.
    async fn predict_batch(&self, individuals: &[Vec<f64>]) -> Result<Option<Vec<f64>>, Status>;

    /// Sends a batch of ground-truth evaluated samples to the surrogate's sliding window.
    async fn ingest_samples(&self, samples: &[(Vec<f64>, f64)]) -> Result<(), Status>;

    /// Fast local check for surrogate readiness.
    async fn is_ready(&self) -> bool;
}
