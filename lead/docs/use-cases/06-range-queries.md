# Use Case: Execute Distributed Range Queries

## Description

Executes ordered lexicographic range queries spanning the distributed cluster. To ensure correct global Hilbert ordering across nodes—where Chord ring partition order (by learned hash) does not match lexicographic Hilbert curve order—LEAD uses a parallel scatter-gather architecture: the receiving coordinator collects local candidate slices, fans out to all known peer nodes, and merge-sorts and deduplicates the combined candidates into a globally sorted top-$N$ slice.

## Primary actor

- External client (e.g. Orchestrator `HilbertNeighborStore` during k-NN searches)

## Supporting systems

- `KeyStore` (`StorageEngine` wrapping `InMemoryStore` or persistent `SledStore`)
- `VirtualNode` routing tables (successor lists and base-10 finger tables)
- `RemoteNode` (`RangeClient` gRPC interface)

## Preconditions

- Keys in `KeyStore` are stored in lexicographical order (e.g. Hilbert-prefixed keys: `{curve_hex}{hex_hilbert}|{canonical_json}`).

## Main steps: Range Query Execution (`handle_range_query`)

1. A client invokes `RangeQuery { start_key, count, caller_address, model_version }`.
2. **Local-Only Sentinel Check**:
   - If `caller_address == "__local__"`:
     - The receiving node computes overscan: $\text{overscan} = \max(\text{count}, \text{count} \times \text{range\_overscan\_multiplier})$.
     - Executes local `storage.range_scan(start_key, overscan)`.
     - Immediately returns `RangeResult { entries: local_entries, complete: true, next_address: "" }` without triggering another scatter-gather round.
3. **Scatter-Gather Coordination Path**:
   - When called by an external client:
     1. **Peer Discovery**: The coordinator queries its local virtual nodes' successor lists and finger tables via `known_peer_addresses()`, collecting all unique remote cluster peer URIs.
     2. **Local Scan**: Performs local `storage.range_scan(start_key, overscan)` to retrieve this node's candidate slice.
     3. **Scatter Fan-Out**: Dispatches `range_query` requests with `caller = "__local__"` to all discovered peer URIs.
     4. **Gather & Merge**: Aggregates all returned candidate slices with the local entries.
     5. **Global Hilbert Sort**: Sorts the combined candidates lexicographically by key (`all.sort_unstable_by(|(a, _), (b, _)| a.cmp(b))`), matching the true Hilbert space-filling curve ordering.
     6. **Deduplication & Truncation**: Deduplicates entries by key (`all.dedup_by`) and truncates to the requested `count`.
     7. Returns `RangeResult { entries, complete: true, next_address: "" }`.

## Postconditions

- Up to `count` contiguous, globally ordered key-value pairs are returned in true Hilbert lexicographic sequence.
- High-dimensional nearest neighbors mapped via Hilbert curve embeddings are accurately retrieved regardless of how virtual nodes are distributed across the Chord ring.

## Failure cases

- Peer node unreachable or times out during scatter fan-out: The coordinator logs a warning, skips the unreachable peer's entries, and returns the sorted merge of available peer and local candidates.
