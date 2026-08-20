# Orchestrator Use Cases

Detailed functional use-case specifications describing interactions, invariants, and failure semantics across all subsystems.

## Index of Use Cases

1. **[`01-startup.md`](01-startup.md)**: Bootstrapping and subsystem wiring.
2. **[`02-configuration.md`](02-configuration.md)**: Runtime configuration loading and environment overrides.
3. **[`03-transport.md`](03-transport.md)**: Channel initialization, readiness polling, and pooling.
4. **[`04-ga-run.md`](04-ga-run.md)**: Generational evolutionary optimization loop.
5. **[`05-evaluation.md`](05-evaluation.md)**: Worker simulator batch evaluation over gRPC.
6. **[`06-gene-cache.md`](06-gene-cache.md)**: In-memory exact caching and distance calculations.
7. **[`07-lead-store.md`](07-lead-store.md)**: Storing and retrieving genes via the LEAD Chord DHT.
8. **[`08-eviction.md`](08-eviction.md)**: Evicting stale gene records across generation boundaries.
9. **[`09-ga-operators.md`](09-ga-operators.md)**: Population initialization, elite survivor selection, and mutation.
10. **[`10-ring-membership.md`](10-ring-membership.md)**: Chord stabilization, join protocol, and successor maintenance.
11. **[`11-migration.md`](11-migration.md)**: Island-model migrant exchange across the ring.
12. **[`12-neighbor-store.md`](12-neighbor-store.md)**: Multi-probe Hilbert curve nearest neighbor search.
13. **[`13-surrogate-client.md`](13-surrogate-client.md)**: Querying and feeding the external surrogate service.
14. **[`14-multi-tier-evaluation.md`](14-multi-tier-evaluation.md)**: Multi-Tier ($\epsilon$-Bypass) hierarchical evaluation pipeline.
