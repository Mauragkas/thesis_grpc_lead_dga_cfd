# Use Case: Ingest Ground-Truth Samples into Sliding Window

## Description

Ingests newly evaluated ground-truth $(\mathbf{x}^*, y_{\text{true}})$ pairs from the Tier 3 worker simulator into the surrogate's FIFO sliding window buffer and checks if background retraining should be triggered.

## Primary Actor

- Orchestrator `MultiTierEvaluator` / `GrpcSurrogateClient`

## Supporting Systems

- `SurrogateServer`
- `SurrogateState`
- `SlidingWindowBuffer`

## Main Steps

1. Client sends `IngestSamplesRequest` with evaluated `Sample` objects over gRPC.
2. `SurrogateServer` locks the `SlidingWindowBuffer`.
3. For each sample:
   - Validate dimension against existing buffer dimension.
   - If buffer is at capacity (`WINDOW_SIZE`), evict the oldest entry from the front.
   - Push new sample to the back.
4. Increment `samples_since_retrain` atomic counter.
5. Check training condition:
   - If (`samples_since_retrain >= RETRAIN_INTERVAL` or `trigger_training = true`) AND `current_window_size >= MIN_TRAIN_SAMPLES`, trigger asynchronous background retraining.
6. Return `IngestSamplesResponse` containing `current_window_size` and `training_started` flag.

## Postconditions

- Sliding window buffer is updated with recent high-fitness designs.
- Retraining is triggered if sample accumulation threshold is met.
