# Orchestrator Crate

The `orchestrator` crate is the distributed Genetic Algorithm (GA) driver for 3D-printable aircraft design optimization. It executes the island model over a self-stabilizing Chord-like ring, performs hierarchical multi-tier evaluations ($\epsilon$-bypass cache hits, fast external MLP surrogate predictions, and true worker simulations), persists individuals to the LEAD DHT via multi-probe Hilbert curve indexing, enforces transport resiliency (circuit breaker and retries), supports dynamic convergence with stagnation-based early stopping, and returns full candidate design genomes (`GaResult`).

## Architecture

- **`src/ga/`**: Core GA runner (`GaRunner`), generational lifecycle, `ProgressTracker` for early stopping, and pure functional operators (`random_population`, `select_survivors`, `next_generation`).
- **`src/evaluator/`**: Multi-Tier ($\epsilon$-Bypass) evaluation pipeline, Tier metrics tracking, worker simulator client (`GrpcEvaluator`), `CircuitBreaker`, and exponential backoff retry policy.
- **`src/surrogate_client/`**: gRPC client adapter for the external `surrogate_node` microservice.
- **`src/ring/`**: Chord-like ring membership, SHA-256 address hashing, and background stabilization loop.
- **`src/migration/`**: Island-model migration, top-K selection, and incoming migrant buffering (`MigrantBuffer`).
- **`src/gene_store/`**: Local in-memory exact cache with generation-age eviction (`InMemoryGeneStore`).
- **`src/neighbor_store/`**: Multi-probe nearest-neighbor store mapping continuous gene vectors to the LEAD DHT (`HilbertNeighborStore`).
- **`src/lead_store/`**: gRPC client for the LEAD Chord DHT (`GrpcLeadStore`).
- **`src/transport/`**: Tonic transport helpers (endpoint construction, channel readiness, and connection pooling).
- **`src/hilbert.rs`**: Multi-probe Compact Hilbert curve embedding delegating to the unified `hilbert_rs` crate.
- **`src/config.rs`**: Environment configuration parsing for all 8 subsystem config structs.

## Key Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `EVAL_ENDPOINT` | `load-balancer:50051` | gRPC endpoint of Envoy or evaluation worker pool |
| `GA_SEED` | `42` | PRNG seed for deterministic runs |
| `MAX_GENERATIONS` / `GA_GENERATIONS` | `100` | Maximum generations to run (safety upper bound) |
| `MIN_GENERATIONS` / `GA_MIN_GENERATIONS` | `10` | Minimum generations before allowing early stopping |
| `STAGNATION_PATIENCE` / `GA_STAGNATION_PATIENCE` | `10` | Stagnant generations before early stopping (0 = disabled) |
| `MIN_IMPROVEMENT` / `GA_MIN_IMPROVEMENT` | `0.001` | Minimum fitness improvement to reset stagnation counter |
| `LEAD_ENDPOINT` | `None` | Optional LEAD DHT endpoint for distributed spatial caching |
| `SURROGATE_ENDPOINT` | `None` | Optional `surrogate_node` microservice endpoint |
| `TIER_EPSILON_EXACT` | `0.005` | Distance threshold for Tier 1 exact cache hits |
| `TIER_RADIUS_R` | `0.15` | Distance threshold for Tier 2 MLP surrogate interpolation |
| `TIER_K_NEIGHBORS` | `15` | Nearest neighbors retrieved from LEAD DHT for Tier 2 |
| `RING_BIND` | `0.0.0.0:50052` | Socket address for Ring gRPC server |
| `RING_SELF_ADDRESS` | `orchestrator:50052` | Public address advertised to ring peers |
| `RING_BOOTSTRAP` | `None` | Seed orchestrator address to join an existing ring |
| `MIGRATION_INTERVAL` | `5` | Generations between island migrations |
| `MIGRATION_COUNT` | `3` | Number of top individuals emigrated per interval |
| `GENE_STORE_MAX_AGE` | `5` | Generation TTL for cached records |
| `LOG_FILE_PATH` / `LOG_DIR` | `None` | Path / directory for structured JSON file logging |

## Build & Test

```bash
# Build binary and library
cargo build --release

# Run unit and integration tests
cargo test

# Check code formatting and lints
cargo clippy --all-targets
```

For detailed architectural specifications, domain models, sequence diagrams, and use cases, see [`docs/README.md`](docs/README.md).
