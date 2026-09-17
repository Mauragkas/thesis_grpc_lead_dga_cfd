# Use Case: Load Configuration from Environment

## Description

Reads runtime overrides from environment variables and combines them with defaults for the GA, transport, gene store, LEAD DHT, Ring topology, and Island migration configurations.

## Primary actor

- Deployment environment / Orchestrator startup

## Environment variables

### Genetic Algorithm & Evaluator
- `EVAL_ENDPOINT`: gRPC address of the evaluator load balancer (default: `"load-balancer:50051"`).
- `GA_SEED`: RNG seed for reproducible genetic algorithm runs (default: `42`).

### In-Memory Cache
- `GENE_STORE_MAX_AGE`: Maximum generation TTL for cached gene records before eviction (default: `5`).

### Distributed Spatial DHT & Surrogate
- `LEAD_ENDPOINT`: Optional gRPC address of a LEAD DHT node.
- `SURROGATE_ENDPOINT`: Optional gRPC address of the surrogate node microservice (e.g. `"surrogate:50054"`).

### Multi-Tier Evaluation Pipeline ($\epsilon$-Bypass)
- `TIER_EPSILON_EXACT`: Distance threshold $d_{\min} < \epsilon_{\text{exact}}$ for Tier 1 0-FLOP cache hits (default: `0.005`).
- `TIER_RADIUS_R`: Distance threshold $\epsilon_{\text{exact}} \le d_{\min} \le R$ for Tier 2 MLP surrogate interpolation (default: `0.15`).
- `TIER_K_NEIGHBORS`: Number of nearest neighbors queried in LEAD DHT (default: `15`).

### Ring Topology & Migration
- `RING_BIND`: Local bind socket address for the Ring gRPC server (default: `"0.0.0.0:50052"`).
- `RING_SELF_ADDRESS`: Public address of this orchestrator advertised to ring peers (default: `"orchestrator:50052"`).
- `RING_BOOTSTRAP`: Optional seed/bootstrap orchestrator address to join the ring.
- `MIGRATION_INTERVAL`: Number of generations between island migrations (default: `5`).
- `MIGRATION_COUNT`: Number of top individuals to emigrate per interval (default: `3`).

### Transport Resiliency
- `CIRCUIT_BREAKER_FAILURE_THRESHOLD`: Consecutive failures tripping circuit breaker from Closed to Open (default: `5`).
- `CIRCUIT_BREAKER_RECOVERY_TIMEOUT_MS`: Cooling duration in milliseconds before probing HalfOpen state (default: `1500`).
- `MAX_INDIVIDUAL_RETRIES`: Maximum retries per individual evaluation attempt (default: `5`).
- `FALLBACK_PENALTY_ON_EXHAUSTION`: Whether to assign penalty fitness `-1e9` upon retries exhaustion instead of failing GA run (default: `false`).

## Main steps

1. Start with default `GaConfig`, overriding `eval_endpoint` and `seed` if `EVAL_ENDPOINT` or `GA_SEED` are set.
2. Start with default `TransportConfig`, overriding circuit breaker thresholds (`CIRCUIT_BREAKER_FAILURE_THRESHOLD`, `CIRCUIT_BREAKER_RECOVERY_TIMEOUT_MS`), retries (`MAX_INDIVIDUAL_RETRIES`), and fallback flag (`FALLBACK_PENALTY_ON_EXHAUSTION`) if set.
3. Start with default `GeneStoreConfig`, overriding `max_age_generations` if `GENE_STORE_MAX_AGE` is set.
4. Start with default `LeadConfig`, setting `endpoint` if `LEAD_ENDPOINT` is set.
5. Start with default `SurrogateClientConfig`, setting `endpoint` if `SURROGATE_ENDPOINT` is set.
6. Start with default `TierConfig`, overriding `epsilon_exact`, `radius_r`, or `k_neighbors` if set.
7. Start with default `RingConfig`, overriding `bind_address`, `self_address`, or `bootstrap_address` if set.
8. Start with default `MigrationConfig`, overriding `interval_generations` or `migrant_count` if set.
9. Return `(GaConfig, TransportConfig, GeneStoreConfig, LeadConfig, SurrogateClientConfig, TierConfig, RingConfig, MigrationConfig)`.

## Postconditions

- A complete set of 8 runtime configuration structs is available for all orchestrator subsystems.

## Notes

- Invalid numeric environment values (e.g. non-numeric string for `GA_SEED` or `MIGRATION_INTERVAL`) log a warning and retain their default values.
- `TransportConfig` contains default gRPC deadlines, retry parameters, keepalive timeouts, and circuit breaker settings.

