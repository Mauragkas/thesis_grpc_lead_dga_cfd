# Surrogate Node Sequence Diagrams

Visual sequence diagrams capturing interaction workflows across surrogate node subsystems, gRPC transport, and background training concurrency.

## Diagrams

- **`startup/`**: Service initialization, environment variable parsing, buffer allocation, and Tonic server listener binding.
- **`prediction/`**: Handling `PredictBatch` requests with non-blocking atomic reads of the active `MlpSurrogate` model.
- **`ingestion-retraining/`**: True sample ingestion into the sliding window FIFO buffer, threshold-triggered background training, and zero-downtime atomic model swap.
- **`cli-benchmark/`**: Offline model comparison CLI workflow across GP, RF, k-NN, and MLP models.
