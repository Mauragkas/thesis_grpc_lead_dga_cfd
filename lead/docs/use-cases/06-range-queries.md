# Use Case: Execute Distributed Range Queries

## Description

Executes ordered lexicographic range queries spanning multiple nodes across the ring. Uses the RMI LearnedHASH model to identify the partition start, executes local overscanned scans, filters for legitimately owned keys, and forwards along the successor chain until the requested count is met.

## Primary actor

- External client (e.g. Orchestrator `HilbertNeighborStore` during k-NN searches)

## Supporting systems

- `KeyStore` (`InMemoryStore` BTreeMap)
- `LearnedIndex` / `RmiModel`
- `RemoteNode` (`RangeClient` gRPC interface)

## Preconditions

- Keys in `InMemoryStore` are stored in lexicographical order (e.g. Hilbert-prefixed keys).

## Main steps: Initial Range Query (`RangeQuery`)

1. A client invokes `RangeQuery { start_key, count, caller_address, model_version }`.
2. The handling node fetches the specified or active `RmiModel` and predicts the starting hash: $id = \text{model.predict}(start\_key)$.
3. The node identifies the local `VirtualNode` responsible for $id$.
4. **Local Scan**: The node performs an overscanned range scan (`count * 3`) in `InMemoryStore` starting at `start_key`.
5. **Ownership Filter**: `collect_owned()` filters the scanned keys to include only keys owned by this node under the model.
6. **Completion Check**:
   - If accumulated entries $\ge count$: Return `RangeResponse { entries, complete: true }`.
   - If entries $< count$: Forward the remaining request via `RangeForward` to the vnode's successor.

## Main steps: Range Forwarding (`RangeForward`)

1. When a node receives `RangeForward { from_key, count, origin_vid, payload, ... }`:
2. Find the local vnode whose range follows the predicted ID of `from_key`.
3. Perform `range_scan_after(from_key, overscan)` on local storage.
4. Filter for owned keys and append to `payload`.
5. If `payload.len() >= count` or the successor reaches `origin_vid` (full ring wrap):
   - Return `RangeResponse { entries: payload, complete: true }`.
6. Otherwise, forward to the next successor with updated `last_key` and remaining count.

## Postconditions

- Up to `count` contiguous ordered key-value pairs are collected and returned to the caller.
- Hilbert-encoded multi-dimensional range searches correctly recover proximate nearest neighbors.

## Failure cases

- Successor node failure during forwarding: Returns current accumulated partial entries with `complete: true`.
