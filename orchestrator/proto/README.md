# Protocol Buffer Definitions

Proto definitions consumed by `orchestrator/build.rs` to generate client and server stubs at compile time using `tonic-build`.

## Services

- **`eval.proto`**: `eval.Evaluator` service definition (`EvaluateBatch`).
- **`lead.proto`**: `lead.Lead` service definition for DHT storage, retrieval, and range queries.
- **`ring.proto`**: `orchestrator_ring.Ring` service definition for Chord stabilization, node discovery, and island migrations.
- **`surrogate.proto`**: `surrogate.SurrogateService` definition for batch predictions (`PredictBatch`), true sample ingestion (`IngestSamples`), online training (`Train`), and model health (`GetStatus`).
