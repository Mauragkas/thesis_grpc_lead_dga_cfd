# Use Case: Surrogate Service Startup

## Description

Initializes hardware probes, loads runtime configuration from environment variables, sets up the shared thread-safe sliding window buffer and atomic state, and starts the Tonic gRPC server.

## Primary Actor

- `surrogate_node::main`

## Supporting Systems

- `SurrogateConfig`
- `SurrogateState`
- `SurrogateServer`
- Tonic gRPC Server framework

## Main Steps

1. Initialize JSON tracing logging subscriber.
2. Read runtime parameters from environment (`GRPC_BIND`, `WINDOW_SIZE`, `RETRAIN_INTERVAL`, `MIN_TRAIN_SAMPLES`, `MLP_EPOCHS`, `MLP_LR`, `MLP_BATCH_SIZE`, `MLP_HIDDEN`).
3. Parse socket address (default `0.0.0.0:50054`).
4. Allocate `SurrogateState` with a `SlidingWindowBuffer` of max capacity `WINDOW_SIZE`.
5. Wrap in `SurrogateServer` and construct `SurrogateServiceServer`.
6. Start Tonic server listener and begin serving incoming gRPC calls.

## Postconditions

- gRPC service is active and responsive to `GetStatus`, `PredictBatch`, and `IngestSamples` calls.
- Initial model state is `ready = false` (cold start awaiting sample ingestion).
