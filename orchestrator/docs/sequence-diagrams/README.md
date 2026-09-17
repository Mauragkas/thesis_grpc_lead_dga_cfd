# Orchestrator Sequence Diagrams

Visual sequence diagrams capturing interaction workflows across orchestrator subsystems and remote services.

## Diagrams

- **`startup/`**: Process startup, loading 8 configuration structs, ring server spawning, stabilizer initialization, evaluator connection with circuit breaker and retry policy, neighbor store setup, surrogate client setup, multi-tier evaluator construction, and migration hook creation.
- **`config/`**: Environment variable parsing and defaults fallback for all 8 configuration structs (including surrogate, tier, and transport resilience parameters).
- **`transport/`**: Endpoint construction and channel readiness polling.
- **`ga-run/`**: Full generational loop (immigrant draining, multi-tier evaluation, eviction, island emigration, breeding).
- **`multi-tier/`**: Hierarchical 3-tier ($\epsilon$-bypass) evaluation routing: Tier 1 0-FLOP cache hit, Tier 2 external MLP surrogate prediction, Tier 3 worker simulator evaluation, and active learning feedback to LEAD DHT and surrogate sliding window.
- **`surrogate-client/`**: gRPC interaction with the external `surrogate_node` microservice for batch prediction and sliding window sample ingestion.
- **`evaluation/`**: Worker simulator evaluation chunking via `tokio::spawn`, batch execution guarded by `CircuitBreaker`, and automatic graceful degradation to per-individual retries with exponential backoff and jitter upon transient error.
- **`gene-cache/`**: In-memory exact cache lookup and miss filtering.
- **`lead-store/`**: Routed `PutRouted`, `GetRouted`, and `RangeQuery` calls to the LEAD DHT with 5-second timeout enforcement.
- **`eviction/`**: Generation-age cache eviction lifecycle.
- **`ga-operators/`**: Pure genetic operator steps (population creation, survivor sorting, Gaussian mutation).
- **`ring/`**: Bootstrap join, periodic Chord stabilization (`GetPredecessor`, `Notify`), and successor list refresh.
- **`migration/`**: Island emigration via gRPC `Migrate`, predecessor buffering in `MigrantBuffer`, and generation-start immigrant draining.
- **`neighbor-store/`**: Multi-probe Hilbert key generation (3 rotated curves from unified `hilbert_rs`), concurrent fanout across all curves via `try_join_all` and `join_all`, candidate deduplication, and k-NN query Euclidean re-ranking.
