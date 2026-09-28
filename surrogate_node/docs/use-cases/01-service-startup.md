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

1. Initialize dual logging via `tracing_subscriber`: human-readable plain text on stdout for `docker logs`, and an optional structured JSON file layer for Fluent-Bit log forwarding when `LOG_FILE_PATH` or `LOG_DIR` is configured.
2. Read runtime parameters from environment variables (supporting both standard and `SURROGATE_*` prefixes):
   - `GRPC_BIND` / `SURROGATE_GRPC_BIND`: Socket address (default: `"0.0.0.0:50054"`).
   - `WINDOW_SIZE` / `SURROGATE_WINDOW_SIZE`: FIFO sample buffer capacity (default: `1000`).
   - `RETRAIN_INTERVAL` / `SURROGATE_RETRAIN_INTERVAL`: Newly ingested sample threshold triggering retraining (default: `20`).
   - `MIN_TRAIN_SAMPLES` / `SURROGATE_MIN_TRAIN_SAMPLES`: Minimum samples in buffer before online training starts (default: `100`).
   - `MLP_EPOCHS` / `SURROGATE_MLP_EPOCHS`: Training epochs for backpropagation (default: `250`).
   - `MLP_LR` / `SURROGATE_MLP_LR`: Optimizer learning rate (default: `0.001`).
   - `MLP_BATCH_SIZE` / `SURROGATE_MLP_BATCH_SIZE`: Mini-batch size (default: `32`).
   - `MLP_HIDDEN` / `SURROGATE_MLP_HIDDEN`: Comma-separated hidden layer neuron counts (default: `64, 32`).
   - Activation function defaults to SiLU (`MlpActivation::Silu`).
3. Parse socket address and initialize `SurrogateState` with a `SlidingWindowBuffer` of max capacity `WINDOW_SIZE`.
4. Wrap `SurrogateState` in `SurrogateServer` and construct Tonic `SurrogateServiceServer`.
5. Start Tonic server listener and begin serving incoming gRPC calls.

## Postconditions

- gRPC service is active and responsive to `GetStatus`, `PredictBatch`, and `IngestSamples` calls.
- Initial model state is `ready = false` (cold start awaiting sample ingestion).
