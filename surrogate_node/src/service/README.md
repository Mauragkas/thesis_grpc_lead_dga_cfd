# Surrogate Microservice Module

Implements the standalone gRPC microservice providing online model serving, sliding window sample retention, and background retraining.

## Components

### 1. `buffer.rs` — `SlidingWindowBuffer`
- **Single Responsibility**: Manages a thread-safe, bounded FIFO sliding window buffer of `(genes, fitness)` samples.
- **Eviction**: Automatically drops the oldest sample once `capacity` (e.g. 1000) is exceeded, keeping the training distribution localized to the current GA search manifold.
- **Dataset Extraction**: Flattens stored records into contiguous row-major matrix arrays `(X_matrix, Y_vector, num_samples, dimension)` ready for C++/CUDA training.

---

### 2. `trainer.rs` — `SurrogateTrainer`
- **Single Responsibility**: Trains an `MlpSurrogate` model on a dataset snapshot.
- **Holdout Validation**: Automatically creates an 80/20 train/validation split when $N \ge 15$ and computes evaluation metrics ($R^2$, RMSE, MAE).

---

### 3. `state.rs` — `SurrogateState`
- **Single Responsibility**: Shared coordination state:
  - Owns the `SlidingWindowBuffer` behind a `tokio::sync::Mutex`.
  - Maintains the active model inside an `Arc<tokio::sync::RwLock<Option<Arc<MlpSurrogate>>>>`.
  - Triggers asynchronous background retraining in `tokio::task::spawn_blocking` upon reaching sample intervals (`RETRAIN_INTERVAL`).
  - Executes **zero-downtime atomic model swapping**: prediction requests continue against the current active model while the new model trains in the background.

---

### 4. `server.rs` — `SurrogateServer`
- **Single Responsibility**: Implements the Tonic `SurrogateService` gRPC server trait.
- Maps protobuf requests (`Predict`, `PredictBatch`, `IngestSamples`, `Train`, `GetStatus`) to `SurrogateState` operations with status codes and logging.

---

### 5. `config.rs` — `SurrogateConfig`
- **Single Responsibility**: Encapsulates runtime environment configuration:
  - `grpc_bind`, `window_size`, `retrain_interval`, `min_train_samples`, `mlp_epochs`, `mlp_lr`, `mlp_batch_size`, `mlp_hidden_layers`.
