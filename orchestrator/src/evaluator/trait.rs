use crate::evaluator::tier_metrics::TierMetricsSnapshot;
use tonic::Status;

/// ISP: a single, narrow capability — turning genes into fitnesses.
/// OCP/DIP: new evaluators (mock, local, alternative transports) can be
/// added by implementing this trait without modifying GA code.
#[async_trait::async_trait]
pub trait Evaluator: Send + Sync {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status>;

    /// Optional telemetry snapshot of multi-tier evaluation bypass and hit counts.
    fn tier_metrics(&self) -> Option<TierMetricsSnapshot> {
        None
    }
}

#[async_trait::async_trait]
impl<T: Evaluator + ?Sized> Evaluator for std::sync::Arc<T> {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
        (**self).evaluate_population(population).await
    }

    fn tier_metrics(&self) -> Option<TierMetricsSnapshot> {
        (**self).tier_metrics()
    }
}
