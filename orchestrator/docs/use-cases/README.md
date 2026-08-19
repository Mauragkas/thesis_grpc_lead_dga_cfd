# Orchestrator Use Cases

This folder documents the runtime use cases implemented by the `orchestrator` crate.

## Covered use cases

- [01. Start orchestrator run](01-startup.md)
- [02. Load configuration from environment](02-configuration.md)
- [03. Build transport endpoint and wait for readiness](03-transport.md)
- [04. Run the genetic algorithm loop](04-ga-run.md)
- [05. Evaluate a population through gRPC](05-evaluation.md)
- [06. Short-circuit evaluation with exact gene cache](06-gene-cache.md)
- [07. Persist evaluated genes to LEAD DHT](07-lead-store.md)
- [08. Evict expired gene records](08-eviction.md)
- [09. GA operators (population, selection, breeding)](09-ga-operators.md)
- [10. Manage ring topology and Chord stabilization](10-ring-membership.md)
- [11. Island-model migration (emigration and immigrant integration)](11-migration.md)
- [12. Multi-probe Hilbert neighbor store (store and k-NN query)](12-neighbor-store.md)

