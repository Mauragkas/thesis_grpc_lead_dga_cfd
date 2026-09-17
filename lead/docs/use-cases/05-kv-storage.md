# Use Case: Key-Value Storage (Local and Routed)

## Description

Provides point key-value storage operations. Clients can execute operations either directly on local node storage or through routed DHT calls that use the RMI LearnedHASH model to locate the responsible node.

## Primary actor

- External client (e.g. Orchestrator `GrpcLeadStore`, HTTP client)

## Supporting systems

- `KeyStore` (`StorageEngine` dispatching to `InMemoryStore` or persistent `SledStore`)
- `LearnedIndex` (predicts target hash ID)
- `RemoteNode` (`GrpcRemote`)

## Main steps: Local Key-Value Operations

1. **`GetLocal(key)`**: Checks local `KeyStore` storage engine. Returns value if found, or `NotFound`.
2. **`PutLocal(key, value)`**: Writes `(key, value)` to local `KeyStore` storage engine (durable on disk if Sled is enabled).
   - Calls `record_insertion(key)` on `LearnedIndex` to track drift and trigger PID tuning.
3. **`DeleteLocal(key)`**: Deletes `key` from local `KeyStore` storage engine.

## Main steps: Routed Key-Value Operations

1. **`PutRouted(key, value)`**:
   - Computes target hash ID: $id = \text{learned\_hash}(key)$.
   - Performs Chord lookup `find_successor(id)` starting from the best local vnode.
   - If the responsible node is local: inserts into local storage engine and records insertion.
   - If the responsible node is remote: forwards `PutLocal(key, value)` to the destination node via gRPC.
2. **`GetRouted(key)`**:
   - Computes target hash ID: $id = \text{learned\_hash}(key)$.
   - Resolves the responsible node via `find_successor(id)`.
   - If local: reads from local storage engine.
   - If remote: forwards `GetLocal(key)` to the destination node via gRPC.

## Postconditions

- Keys are stored on the node responsible for their predicted LearnedHASH partition.
- Key insertions feed real-time drift metrics into the learning subsystem.
