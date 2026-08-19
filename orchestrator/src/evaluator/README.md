# Evaluator Module

Defines the worker evaluation contract and gRPC batch evaluation client.

## Files and Code Structure

### 1. `trait.rs`
- **`Evaluator`** (async trait):
  ```rust
  #[async_trait::async_trait]
  pub trait Evaluator: Send + Sync {
      async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, tonic::Status>;
  }
  ```
  Primary port (DIP) used by `GaRunner` to evaluate candidate populations without depending on gRPC transports.

---

### 2. `grpc.rs`
- **`GrpcEvaluator`**:
  - **Fields**:
    - `client: EvaluatorClient<Channel>`: Tonic gRPC client.
    - `cfg: TransportConfig`: Deadlines, retry counts, and backoff durations.
    - `batch_size: usize`: Maximum number of individuals sent per RPC batch.
  - **Methods**:
    - `pub fn new(client: EvaluatorClient<Channel>, cfg: TransportConfig, batch_size: usize) -> Self`: Constructor.
    - `pub async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, tonic::Status>`: Chunks the population into batches of `batch_size` and evaluates them in order, flattening the resulting fitness values.
    - `async fn eval_batch(client: &mut EvaluatorClient<Channel>, cfg: &TransportConfig, chunk: &[Vec<f64>]) -> Result<Vec<f64>, tonic::Status>`: Constructs a `BatchRequest`, invokes `EvaluateBatch` on the remote load balancer, and retries on `Unavailable` status or timeouts up to `cfg.max_attempts`.
