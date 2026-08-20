# Use Case: Offline Model Comparison Benchmark

## Description

Runs an offline comparative benchmark across all 4 surrogate architectures (Gaussian Process with Matérn 5/2 kernel, Random Forest, k-Nearest Neighbors, and Multi-Layer Perceptron) using historical aerodynamic simulation datasets.

## Primary Actor

- User / Developer via CLI `cargo run --bin compare`

## Supporting Systems

- `DatasetLoader`
- `DatasetSplitter`
- `GaussianProcessSurrogate`, `RfSurrogate`, `KnnSurrogate`, `MlpSurrogate`

## Main Steps

1. Load JSON dataset of converged aircraft designs from candidate paths.
2. Extract 11 active aerodynamic design features ($X$) and simulation fitness values ($y$).
3. Partition dataset into 60% Train, 20% Validation, and 20% Holdout Test splits using deterministic seed.
4. Fit all models under identical training partitions.
5. Evaluate holdout test metrics ($R^2$ score, RMSE, MAE, Max Residual).
6. Measure training duration and inference throughput per sample.
7. Print formatted comparative summary table.

## Postconditions

- Evaluates regression accuracy and execution latency across all surrogate models on empirical aerodynamic data.
