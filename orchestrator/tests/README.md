# Orchestrator Integration Tests

Rust integration test suite for the `orchestrator` crate. Tests run against the library facade (`orchestrator::*`).

## Tests

- **`config_test.rs`**: Validates environment variable overrides, defaults, and numeric parsing.
- **`evaluator_test.rs`**: Tests batch evaluation, chunk ordering, and error propagation.
- **`ga_algorithm_test.rs`**: Tests `GaRunner` generational lifecycle, caching, eviction, and error handling.
- **`ga_operators_test.rs`**: Tests population bounds, survivor selection elitism, and Gaussian mutation.
- **`gene_store_test.rs`**: Tests `InMemoryGeneStore` exact lookups, k-NN queries, Euclidean metrics, and TTL eviction.
- **`hilbert_test.rs`**: Tests multi-probe Hilbert key determinism, Skilling algorithm encoding, and nearest neighbor recovery.
- **`migration_test.rs`**: Tests top-K migrant selection.
- **`ring_test.rs`**: Tests SHA-256 hashing, ring interval arithmetic, successor discovery, and predecessor notifications.
- **`transport_test.rs`**: Tests endpoint building and channel readiness polling.

## Running Tests

```bash
cargo test
```
