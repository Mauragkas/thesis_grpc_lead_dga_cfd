# Use Case: Query and Feed External Surrogate Service

## Description

The orchestrator communicates with the external `surrogate_node` microservice over gRPC to obtain fast aerodynamic fitness predictions for candidate designs and feed newly evaluated true designs into the surrogate's online sliding window buffer.

## Primary Actor

- `GrpcSurrogateClient`

## Supporting Systems

- `surrogate.SurrogateService` gRPC daemon (`surrogate-node:50054`)

## Main Steps

### 1. Batch Prediction
1. Build `BatchPredictRequest` containing candidate individuals.
2. Call `PredictBatch` over gRPC.
3. If the surrogate responds with `ready = true`, extract and return predicted fitness values.
4. If `ready = false` (surrogate is still in cold-start with insufficient samples), return `None` so the caller can fall back to ground-truth simulation.
5. If the gRPC call fails or times out, log a warning and return error for fallback.

### 2. Sample Ingestion
1. Filter out rejected or invalid individuals ($y \le -10^8$).
2. Build `IngestSamplesRequest` with $(\mathbf{x}^*, y_{\text{true}})$ pairs.
3. Asynchronously call `IngestSamples` on the surrogate service without blocking the main GA loop.
4. The surrogate node adds samples to its FIFO sliding window buffer and triggers background retraining if sample count thresholds are met.

## Postconditions

- Candidate designs within the interpolation radius receive fast MLP predictions.
- The surrogate service continuously updates its training dataset with ground-truth evaluations from the active search manifold.

## Failure Cases

- Surrogate service offline or unreachable -> Graceful fallback to Tier 3 worker simulator.
- gRPC deadline exceeded -> Fallback to Tier 3 worker simulator.
