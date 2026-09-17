# Use Case: Start Orchestrator Run

## Description

Starts the orchestrator binary, initializes structured JSON logging via `tracing_subscriber`, loads runtime configuration from environment variables, initializes the Chord-like ring member and spawns the Ring gRPC server, optionally joins an existing ring via a bootstrap node and starts the background stabilization loop, connects to the evaluator service (configuring circuit breaker and retry policies), optionally connects to a LEAD DHT node to initialize the multi-probe Hilbert neighbor store, optionally connects to the surrogate node microservice, constructs the multi-tier ($\epsilon$-bypass) evaluator pipeline, configures island-model migration, and begins the genetic algorithm run.

## Primary actor

- System operator / deployment environment

## Supporting systems

- Evaluator service (Envoy load balancer / workers)
- Optional `surrogate_node` microservice (online MLP surrogate)
- Optional LEAD DHT node (distributed spatial index)
- Ring peers (other orchestrator island nodes)
- Environment variables

## Preconditions

- The orchestrator process can start successfully.
- Environment variables may be present to override defaults.
- The evaluator endpoint must be reachable before the GA run begins.

## Main steps

1. The process starts in `main` and delegates to `orchestrator::bootstrap::run()`.
2. Structured JSON logging is initialized via `tracing_subscriber` with target filtering.
3. Configuration is loaded with `config_from_env()`, returning `(GaConfig, TransportConfig, GeneStoreConfig, LeadConfig, SurrogateClientConfig, TierConfig, RingConfig, MigrationConfig)`.
4. The node's address is hashed with `Sha256Hasher` to produce a `NodeInfo` ID; `RingState` and `LocalRingMember` are created.
5. A `MigrantBuffer` and `RingServer` (implementing the tonic `Ring` gRPC service) are created, bound to `ring_cfg.bind_address`, and spawned in a background task.
6. A `GrpcRingClient` and `Stabilizer` are created.
   - If `ring_cfg.bootstrap_address` is present, the stabilizer attempts to join the bootstrap node (retrying up to 10 times with exponential/fixed backoff).
   - If no bootstrap is set, the node starts as the first ring member (successor set to self).
7. The background stabilization loop is spawned (`stabilizer.spawn()`).
8. The evaluator endpoint is converted into a tonic `Endpoint`, and `wait_for_channel()` waits until the channel is ready.
9. A `GrpcEvaluator` is created using the connected channel, configured with `CircuitBreaker` and per-individual `RetryPolicy`.
10. The `InMemoryGeneStore` is created with `EuclideanDistance` and `GenerationEvictor`.
11. If `lead_cfg.endpoint` is present:
    - The LEAD endpoint channel is established via `wait_for_channel()`.
    - A `GrpcLeadStore` client and `HilbertKeyGenerator` (from the unified `hilbert_rs` crate) are created.
    - A `HilbertNeighborStore` is instantiated wrapping the key generator and lead store.
12. If `surrogate_cfg.endpoint` is present:
    - The surrogate node endpoint channel is established via `wait_for_channel()`.
    - A `GrpcSurrogateClient` is instantiated wrapping the Tonic client and transport configuration.
13. `MultiTierEvaluator` is constructed, composing the raw simulator, local gene store, optional LEAD neighbor store, optional surrogate client, and `TierConfig`.
14. `LeadMigration` is instantiated with `TopKSelector`, the local ring member, the ring client, and the migrant buffer.
15. A seeded `StdRng` is initialized from `ga_cfg.seed`.
16. `GaRunner::run()` is executed with `evaluator: &multi_tier_evaluator`.

## Postconditions

- The GA run completes across the configured number of generations.
- Ring membership and periodic stabilization continue running in the background.
- Island migrations are exchanged periodically with ring neighbors.
- The best-ever fitness value is logged and returned.

## Failure cases

- Invalid evaluator URI or evaluator channel readiness deadline exceeded.
- Invalid ring bind address or port collision.
- Bootstrap join retry exhaustion.
- LEAD endpoint invalid or unreachable when configured.
- Unrecoverable gRPC errors during evaluation.

