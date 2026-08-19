# Orchestrator Crate

The `orchestrator` crate is the distributed Genetic Algorithm (GA) driver for aircraft design optimization. It executes the island model over a self-stabilizing Chord-like ring, performs exact local caching and eviction, persists individuals to the LEAD DHT via multi-probe Hilbert curve indexing, and evaluates candidate designs in batches over gRPC via an Envoy load balancer.

## Architecture

- **`src/ga/`**: Core GA runner and pure generational operators.
- **`src/ring/`**: Chord-like ring membership, SHA-256 address hashing, and background stabilization loop.
- **`src/migration/`**: Island-model migration, top-K selection, and incoming migrant buffering.
- **`src/gene_store/`**: Local in-memory exact cache with generation-age eviction.
- **`src/neighbor_store/`**: Multi-probe nearest-neighbor store mapping continuous gene vectors to the LEAD DHT.
- **`src/lead_store/`**: gRPC client for the LEAD Chord DHT.
- **`src/evaluator/`**: gRPC client adapter for worker batch evaluations with retry logic.
- **`src/transport/`**: Tonic transport helpers (endpoint construction, channel readiness).
- **`src/hilbert.rs`**: Multi-probe Compact Hilbert curve embedding (Skilling algorithm).
- **`src/config.rs`**: Environment configuration parsing.

## Build & Test

```bash
cargo build
cargo test
cargo clippy --all-targets
```
