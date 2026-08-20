# Surrogate Client Module

Contains abstractions and gRPC adapters for communicating with the external `surrogate_node` microservice.

## Files and Code Structure

### 1. `trait.rs`
- **`SurrogateClient`** (async trait):
  ```rust
  #[async_trait::async_trait]
  pub trait SurrogateClient: Send + Sync {
      async fn predict_batch(&self, individuals: &[Vec<f64>]) -> Result<Option<Vec<f64>>, tonic::Status>;
      async fn ingest_samples(&self, samples: &[(Vec<f64>, f64)]) -> Result<(), tonic::Status>;
      async fn is_ready(&self) -> bool;
  }
  ```
  Provides a clean abstraction (ISP/DIP) decoupling `MultiTierEvaluator` from concrete gRPC channel details.

---

### 2. `grpc.rs`
- **`GrpcSurrogateClient`**:
  - Implements `SurrogateClient` backed by Tonic `SurrogateServiceClient<Channel>`.
  - Sends batches of individuals to `PredictBatch` and newly evaluated true samples to `IngestSamples`.
  - Caches readiness state to minimize unnecessary RPC calls during cold-start phases.

---

### 3. `mock.rs`
- **`MockSurrogateClient`**:
  - Thread-safe mock implementation for unit testing and offline simulation without running the surrogate gRPC daemon.
