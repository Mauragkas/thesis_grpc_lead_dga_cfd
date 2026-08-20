# Orchestrator Sequence Diagrams

Visual sequence diagrams capturing interaction workflows across orchestrator subsystems and remote services.

## Diagrams

- **`startup/`**: Process startup, config loading, ring server spawning, stabilizer initialization, evaluator connection, neighbor store setup, surrogate client setup, and migration hook creation.
- **`config/`**: Environment variable parsing and defaults fallback for all configuration structs.
- **`transport/`**: Endpoint construction and channel readiness polling.
- **`ga-run/`**: Full generational loop (immigrant draining, multi-tier evaluation, eviction, island emigration, breeding).
- **`multi-tier/`**: Hierarchical 3-tier ($\epsilon$-bypass) evaluation routing: Tier 1 0-FLOP cache hit, Tier 2 external MLP surrogate prediction, Tier 3 worker simulator evaluation, and active learning feedback to LEAD DHT and surrogate sliding window.
- **`surrogate-client/`**: gRPC interaction with the external `surrogate_node` microservice for batch prediction and sliding window sample ingestion.
- **`evaluation/`**: gRPC batch evaluation chunking and timeout retry handling to the worker simulator.
- **`gene-cache/`**: In-memory exact cache lookup and miss filtering.
- **`lead-store/`**: Routed `PutRouted`, `GetRouted`, and `RangeQuery` calls to the LEAD DHT.
- **`eviction/`**: Generation-age cache eviction lifecycle.
- **`ga-operators/`**: Pure genetic operator steps (population creation, survivor sorting, Gaussian mutation).
- **`ring/`**: Bootstrap join, periodic Chord stabilization (`GetPredecessor`, `Notify`), and successor list refresh.
- **`migration/`**: Island emigration via gRPC `Migrate`, predecessor buffering in `MigrantBuffer`, and generation-start immigrant draining.
- **`neighbor-store/`**: Multi-probe Hilbert key generation (3 rotated curves), multi-key storing, and k-NN query with Euclidean re-ranking.
