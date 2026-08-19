# Use Case: Island-Model Migration

## Description

Facilitates genetic diversity exchange between orchestrator nodes running isolated GA islands on the ring. Every `MIGRATION_INTERVAL` generations, top-performing individuals are selected, sent via gRPC to the ring successor, buffered asynchronously, and integrated into the recipient's population.

## Primary actor

- `LeadMigration` / `GaRunner` / `RingServer`

## Supporting systems

- `MigrantSelector` (`TopKSelector`)
- `MigrantBuffer` (thread-safe FIFO queue)
- `RingServer` (`/orchestrator_ring.Ring/Migrate` RPC)
- `LocalRingMember` (provides ring successor destination)

## Preconditions

- Multiple orchestrator nodes are running and form a connected ring.
- `MigrationConfig` is configured (`interval_generations > 0`, `migrant_count > 0`).

## Main steps: Emigration (Sending Migrants)

1. At the end of generation evaluation, `GaRunner` invokes `maybe_emigrate(gen, population, fitnesses)`.
2. If `gen` is a multiple of `interval_generations`:
   1. `TopKSelector` sorts population by fitness descending and picks the top `migrant_count` individuals.
   2. The ring successor is queried from `LocalRingMember`. If successor is `self` (single node), emigration is skipped.
   3. Individuals are mapped into proto `MigrantIndividual` structures.
   4. A gRPC `MigrateRequest` is sent to the successor's `/orchestrator_ring.Ring/Migrate` endpoint.
   5. If the RPC fails, a warning is logged; emigration is best-effort and does not interrupt the GA loop.

## Main steps: Immigration (Receiving and Integrating Migrants)

1. **Inbound RPC**: When `RingServer::migrate` is called by a ring predecessor:
   1. Inbound individuals are parsed into `MigrantIndividual` structs.
   2. Individuals are pushed into `MigrantBuffer`.
   3. An acceptance response is returned.
2. **Integration into GA**: At the start of the next generation in `GaRunner`:
   1. `migration.drain_immigrants()` drains all accumulated migrants from `MigrantBuffer`.
   2. If immigrants are present, they are appended to the local population and the population is truncated to `pop_size`, replacing the lowest slots.

## Postconditions

- Beneficial genetic material propagates across islands along the ring topology.
- Island populations maintain target size `pop_size`.

## Failure cases

- Successor node offline or network partition: emigration logged as warning, local GA continues.
- Buffer overflow: migrant buffer drains every generation, preventing unbounded growth.
