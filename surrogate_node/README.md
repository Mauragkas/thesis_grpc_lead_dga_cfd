# Surrogate Node Crate & Microservice

High-performance aerodynamic surrogate modeling node and gRPC microservice supporting **Multi-Layer Perceptrons (MLP)**, **Gaussian Processes (GP)**, **Random Forests (RF)**, and **k-Nearest Neighbors (k-NN)** with native C++/CUDA/OpenMP acceleration.

## Overview

The `surrogate_node` operates in two primary modes:
1. **gRPC Microservice Daemon (`surrogate_node`)**:
   Runs as an independent service (default port `50054`) maintaining a thread-safe **sliding window buffer** of evaluated designs. It retrains the MLP surrogate model periodically in the background and atomically updates the active model for zero-downtime batch predictions.
2. **Native Compute Library (`surrogate_node` crate)**:
   Provides high-throughput C++/OpenMP/CUDA compute backends for matrix operations, Cholesky factorization, tree ensembles, and backpropagation.

## Architecture

- **`src/service/`**: Microservice layer:
  - `SlidingWindowBuffer`: Thread-safe bounded FIFO sample buffer with automatic staleness eviction.
  - `SurrogateTrainer`: Background training engine for `MlpSurrogate`.
  - `SurrogateState`: Shared state coordinating sample ingestion, asynchronous training triggers, and zero-downtime model swapping.
  - `SurrogateServer`: Tonic gRPC server implementing `SurrogateService`.
  - `SurrogateConfig`: Runtime configuration parsed from environment variables.
- **`src/model/`**: Surrogate model implementations:
  - `MlpSurrogate`: Multi-Layer Perceptron neural network with SiLU/ReLU activations and Adam optimizer.
  - `GaussianProcessSurrogate`: GP regression with Matérn 5/2 / RBF kernels and Cholesky solver.
  - `RfSurrogate`: Multi-threaded Random Forest regressor with tree variance uncertainty.
  - `KnnSurrogate`: Exact Euclidean k-Nearest Neighbors regressor.
- **`src/backend/`**: Hardware dispatchers and C++ FFI bindings (CPU OpenMP, CUDA, ROCm).
- **`src/data/`**: Feature scalers (`StandardScaler`, `TargetScaler`) and dataset loaders/splitters.
- **`native/`**: High-performance C++17 and CUDA engine files (`native/src/` and `native/incl/`).
- **`proto/`**: gRPC protobuf definitions (`proto/surrogate.proto`).

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `GRPC_BIND` / `SURROGATE_GRPC_BIND` | `0.0.0.0:50054` | Socket address for the gRPC server |
| `WINDOW_SIZE` / `SURROGATE_WINDOW_SIZE` | `1000` | Maximum capacity of the sliding window buffer |
| `RETRAIN_INTERVAL` / `SURROGATE_RETRAIN_INTERVAL` | `20` | Number of ingested samples between background retraining cycles |
| `MIN_TRAIN_SAMPLES` / `SURROGATE_MIN_TRAIN_SAMPLES` | `30` | Minimum samples required before marking the model ready |
| `MLP_EPOCHS` / `SURROGATE_MLP_EPOCHS` | `250` | Training epochs per online retraining cycle |
| `MLP_LR` / `SURROGATE_MLP_LR` | `0.001` | Learning rate for the MLP optimizer |
| `MLP_BATCH_SIZE` / `SURROGATE_MLP_BATCH_SIZE` | `32` | Mini-batch size during training |
| `MLP_HIDDEN` / `SURROGATE_MLP_HIDDEN` | `64,32` | Comma-separated hidden layer dimensions |

## Build & Test

```bash
# Build the microservice daemon and library
cargo build --release

# Run the test suite
cargo test

# Run the CLI benchmark against historical aerodynamic datasets
cargo run --bin surrogate_node -- --bench tests/configs_and_scores.json
```
