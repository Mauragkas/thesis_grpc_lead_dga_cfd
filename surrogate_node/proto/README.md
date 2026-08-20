# Surrogate Protocol Buffer Definitions

Defines the gRPC interface for the surrogate node microservice, compiled at build time by `surrogate_node/build.rs` and `orchestrator/build.rs`.

## Service: `SurrogateService`

```protobuf
service SurrogateService {
  rpc Predict (PredictRequest) returns (PredictResponse);
  rpc PredictBatch (BatchPredictRequest) returns (BatchPredictResponse);
  rpc IngestSamples (IngestSamplesRequest) returns (IngestSamplesResponse);
  rpc Train (TrainRequest) returns (TrainResponse);
  rpc GetStatus (StatusRequest) returns (StatusResponse);
}
```

### RPC Endpoints

1. **`PredictBatch(BatchPredictRequest) -> BatchPredictResponse`**:
   - Evaluates a batch of candidate individuals through the active MLP surrogate model.
   - Returns array of predicted fitness values and a `ready` boolean flag indicating whether the model is initialized.

2. **`IngestSamples(IngestSamplesRequest) -> IngestSamplesResponse`**:
   - Ingests newly evaluated ground-truth $(\mathbf{x}^*, y_{\text{true}})$ pairs into the FIFO sliding window buffer.
   - Automatically triggers asynchronous background retraining if the sample threshold (`RETRAIN_INTERVAL`) is reached.

3. **`Train(TrainRequest) -> TrainResponse`**:
   - Manually triggers an asynchronous retraining cycle over the current buffer dataset.
   - Returns validation metrics ($R^2$, RMSE) from the latest training run.

4. **`GetStatus(StatusRequest) -> StatusResponse`**:
   - Returns current buffer size, max capacity, active model version, hardware device name, and readiness flag.
