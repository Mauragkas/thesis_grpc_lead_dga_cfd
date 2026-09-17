# Use Case: Online Background Retraining & Atomic Swap

## Description

Trains a new `MlpSurrogate` model on a snapshot of the current sliding window buffer in an asynchronous background task, and atomically updates the active model upon completion.

## Primary Actor

- `SurrogateState`

## Supporting Systems

- `SurrogateTrainer`
- `MlpSurrogate`
- `DatasetSplitter`
- `tokio::task::spawn_blocking`

## Main Steps

1. `SurrogateState::trigger_train_async(force_train)` is invoked (via threshold detection with `force_train = false` or explicit `Train` RPC with `force_train = true`).
2. Atomically verify `is_training` flag with CAS (`compare_exchange(false, true)`). If already training, exit without starting a duplicate run.
3. Extract dataset snapshot `(X_matrix, Y_vector, num_samples, dim)` from the `SlidingWindowBuffer`.
4. Spawn a background thread via `tokio::task::spawn_blocking`.
5. If $N \ge 15$, split dataset 60/20/20 train/validation/test using `DatasetSplitter`.
6. Fit standard scalers (`StandardScaler`, `TargetScaler`).
7. Train `MlpSurrogate` using native C++/OpenMP/CUDA backpropagation over configured epochs (`MLP_EPOCHS`).
8. Evaluate metrics on validation holdout ($R^2$, RMSE, MAE).
9. Atomically swap the new `MlpSurrogate` into `active_model` (`RwLock<Option<Arc<MlpSurrogate>>>`).
10. Increment `model_version`, record validation metrics in `last_metrics` (`RwLock<Option<EvaluationMetrics>>`), reset `samples_since_retrain`, and clear `is_training` flag.

## Postconditions

- Active model is atomically updated to the newly trained weights.
- Ongoing and future `PredictBatch` calls immediately use the updated model with zero prediction downtime.
