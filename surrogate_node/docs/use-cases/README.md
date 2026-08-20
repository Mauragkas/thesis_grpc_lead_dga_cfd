# Surrogate Node Use Cases

Detailed functional use-case specifications describing interactions, invariants, and failure semantics for the `surrogate_node` microservice.

## Index of Use Cases

1. **[`01-service-startup.md`](01-service-startup.md)**: Server startup, configuration loading, buffer allocation, and gRPC endpoint binding.
2. **[`02-batch-prediction.md`](02-batch-prediction.md)**: Handling `PredictBatch` requests with the active MLP model.
3. **[`03-sample-ingestion.md`](03-sample-ingestion.md)**: Ingesting true evaluated designs into the sliding window FIFO buffer.
4. **[`04-online-retraining.md`](04-online-retraining.md)**: Asynchronous background model fitting and zero-downtime atomic model swap.
5. **[`05-model-comparison.md`](05-model-comparison.md)**: Executing offline comparative benchmarks across GP, RF, k-NN, and MLP models.
