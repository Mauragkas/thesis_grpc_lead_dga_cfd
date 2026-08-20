# Use Case: Batch Fitness Prediction

## Description

Evaluates a batch of candidate aircraft gene vectors using the active Multi-Layer Perceptron surrogate model and returns predicted fitness values.

## Primary Actor

- Orchestrator `GrpcSurrogateClient`

## Supporting Systems

- `SurrogateServer`
- `SurrogateState`
- `MlpSurrogate` (active model instance)
- Native C++/OpenMP/CUDA compute engine

## Main Steps

1. Client sends `BatchPredictRequest` containing candidate individuals over gRPC.
2. `SurrogateServer` delegates to `SurrogateState::predict_batch`.
3. `SurrogateState` checks whether an active model exists:
   - If no model is active (`None`), immediately return `BatchPredictResponse` with `ready = false` and an empty list.
   - If an active model exists, acquire a non-blocking read lock.
4. Flatten the input vectors into a row-major array.
5. Standardize input features using the model's fitted `StandardScaler`.
6. Execute parallel forward pass through the native C++/CUDA MLP backend.
7. Inverse-transform output predictions using `TargetScaler`.
8. Return `BatchPredictResponse` with `fitnesses` and `ready = true`.

## Postconditions

- Candidate designs receive predicted fitness values in sub-milliseconds without blocking ongoing background training.
