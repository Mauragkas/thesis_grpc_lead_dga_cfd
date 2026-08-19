# Orchestrator Sequence Diagrams

Visual sequence diagrams capturing interaction workflows across orchestrator subsystems and remote services.

## Diagrams

- **`startup/`**: Process startup, config loading, ring server spawning, stabilizer initialization, evaluator connection, neighbor store setup, and migration hook creation.
- **`config/`**: Environment variable parsing and defaults fallback for all 6 configuration structs.
- **`transport/`**: Endpoint construction and channel readiness polling.
- **`ga-run/`**: Full generational loop (immigrant draining, exact cache check, batch evaluation, LEAD neighbor store indexing, eviction, island emigration, breeding).
- **`evaluation/`**: gRPC batch evaluation chunking and timeout retry handling.
- **`gene-cache/`**: In-memory exact cache lookup and miss filtering.
- **`lead-store/`**: Routed `PutRouted`, `GetRouted`, and `RangeQuery` calls to the LEAD DHT.
- **`eviction/`**: Generation-age cache eviction lifecycle.
- **`ga-operators/`**: Pure genetic operator steps (population creation, survivor sorting, Gaussian mutation).
- **`ring/`**: Bootstrap join, periodic Chord stabilization (`GetPredecessor`, `Notify`), and successor list refresh.
- **`migration/`**: Island emigration via gRPC `Migrate`, predecessor buffering in `MigrantBuffer`, and generation-start immigrant draining.
- **`neighbor-store/`**: Multi-probe Hilbert key generation (3 rotated curves), multi-key storing, and k-NN query with Euclidean re-ranking.
