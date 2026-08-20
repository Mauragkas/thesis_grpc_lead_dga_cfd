# Surrogate Node Source Code

Source modules for the `surrogate_node` crate.

## Subsystems

- [`service/`](service/README.md): Microservice layer containing the sliding window sample buffer, background trainer, shared atomic state, and gRPC server.
- [`model/`](model/README.md): High-level surrogate models:
  - `MlpSurrogate`: Multi-Layer Perceptron neural network.
  - `GaussianProcessSurrogate`: Gaussian Process with hyperparameter optimization.
  - `RfSurrogate`: Random Forest regressor.
  - `KnnSurrogate`: k-Nearest Neighbors regressor.
- `backend/`: Hardware dispatchers and FFI bindings for CPU OpenMP, CUDA, and ROCm.
- `data/`: Standard scalers (`StandardScaler`, `TargetScaler`) and dataset splitters.
- `domain/`: Domain structs (`AircraftConfig`, `DatasetRecord`, `EvaluationMetrics`, `GpHyperparameters`).
- `proto.rs`: Protobuf bindings generated from `proto/surrogate.proto`.

## Entry Points

- **`src/main.rs`**: Starts the Tonic gRPC daemon listening on `GRPC_BIND` (default `0.0.0.0:50054`), with optional CLI benchmark mode (`--bench <file.json>`).
- **`src/bin/compare.rs`**: Multi-model comparison CLI benchmark evaluating GP, RF, k-NN, and MLP models across training/validation/test holdouts.
