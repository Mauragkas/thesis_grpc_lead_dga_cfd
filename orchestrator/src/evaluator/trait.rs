use tonic::Status;

/// ISP: a single, narrow capability — turning genes into fitnesses.
/// OCP/DIP: new evaluators (mock, local, alternative transports) can be
/// added by implementing this trait without modifying GA code.
#[async_trait::async_trait]
pub trait Evaluator: Send + Sync {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status>;
}
