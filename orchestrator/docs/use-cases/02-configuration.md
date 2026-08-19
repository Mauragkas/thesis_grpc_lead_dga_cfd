# Use Case: Load Configuration from Environment

## Description

Reads runtime overrides from environment variables and combines them with defaults for the GA, transport, gene store, LEAD DHT, Ring topology, and Island migration configurations.

## Primary actor

- Deployment environment / Orchestrator startup

## Environment variables

- `EVAL_ENDPOINT`: gRPC address of the evaluator load balancer (default: `"load-balancer:50051"`).
- `GA_SEED`: RNG seed for reproducible genetic algorithm runs (default: `42`).
- `GENE_STORE_MAX_AGE`: Maximum generation TTL for cached gene records (default: `5`).
- `LEAD_ENDPOINT`: Optional gRPC address of a LEAD DHT node.
- `RING_BIND`: Local bind socket address for the Ring gRPC server (default: `"0.0.0.0:50052"`).
- `RING_SELF_ADDRESS`: Public address of this orchestrator advertised to ring peers (default: `"orchestrator:50052"`).
- `RING_BOOTSTRAP`: Optional seed/bootstrap orchestrator address to join the ring.
- `MIGRATION_INTERVAL`: Number of generations between island migrations (default: `5`).
- `MIGRATION_COUNT`: Number of top individuals to emigrate per interval (default: `3`).

## Main steps

1. Start with default `GaConfig`.
2. Override `eval_endpoint` if `EVAL_ENDPOINT` is set.
3. Override `seed` if `GA_SEED` parses successfully as `u64`.
4. Start with default `GeneStoreConfig`.
5. Override `max_age_generations` if `GENE_STORE_MAX_AGE` parses successfully as `usize`.
6. Start with default `LeadConfig`.
7. Set `endpoint` if `LEAD_ENDPOINT` is set.
8. Start with default `RingConfig`.
9. Override `bind_address` if `RING_BIND` is set.
10. Override `self_address` if `RING_SELF_ADDRESS` is set.
11. Set `bootstrap_address` if `RING_BOOTSTRAP` is set.
12. Start with default `MigrationConfig`.
13. Override `interval_generations` if `MIGRATION_INTERVAL` parses as `usize`.
14. Override `migrant_count` if `MIGRATION_COUNT` parses as `usize`.
15. Return `(GaConfig, TransportConfig, GeneStoreConfig, LeadConfig, RingConfig, MigrationConfig)`.

## Postconditions

- A complete set of 6 runtime configuration structs is available for the orchestrator subsystems.

## Notes

- Invalid numeric environment values (e.g. non-numeric string for `GA_SEED` or `MIGRATION_INTERVAL`) log a warning and retain their default values.
- `TransportConfig` contains default gRPC deadlines, retry parameters, and keepalive timeouts.

