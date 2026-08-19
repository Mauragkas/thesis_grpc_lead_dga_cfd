# Use Case: Evict Expired Gene Records

## Description

Removes gene records that have not been accessed within the configured generation-age threshold.

## Primary actor

- `InMemoryGeneStore`

## Supporting systems

- `EvictionPolicy`
- `GenerationEvictor`

## Main steps

1. The GA run completes a generation.
2. `evict_expired(current_generation)` is called.
3. The store asks the eviction policy whether each record should be removed.
4. Records whose `last_accessed_gen` is too old are removed.

## Postconditions

- Stale records are removed from memory.

## Notes

- Current policy: `current_generation - last_accessed_gen > max_age`
- Accessing a record refreshes its TTL.
