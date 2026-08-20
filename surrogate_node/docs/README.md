# Surrogate Node Documentation

Comprehensive architectural, functional, and sequence documentation for the `surrogate_node` crate and microservice.

## Documentation Structure

- [`use-cases/`](use-cases/README.md): Detailed use case specifications describing the gRPC microservice operations, buffer management, online retraining, and benchmark execution.
- [`sequence-diagrams/`](sequence-diagrams/README.md): Mermaid sequence diagrams visualizing request flows, background training concurrency, and atomic model swapping.

## Core Concepts

1. **Sliding Window Buffer**: Bounded FIFO queue of evaluated aerodynamic points that automatically evicts stale configurations as the GA population drifts toward optimal regimes.
2. **Online MLP Retraining**: Background asynchronous training loop powered by native C++/OpenMP/CUDA backpropagation without blocking active prediction traffic.
3. **Zero-Downtime Model Swapping**: Atomic pointer exchange of the active `MlpSurrogate` model upon training completion.
4. **Multi-Model Native Engine**: Unified C++/CUDA acceleration for Gaussian Processes, Random Forests, k-Nearest Neighbors, and Multi-Layer Perceptrons.
