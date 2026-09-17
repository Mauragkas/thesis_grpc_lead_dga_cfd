# LEAD DHT Node Crate

The `lead-node` crate implements the LEAD Chord-like Distributed Hash Table with online learning, Two-Stage Recursive Model Indexing (RMI), online PID tuning, and virtual nodes.

## Architecture

- **`src/lead/`**: Physical `LeadNode` coordinator, `VirtualNode` vnode management, and lookup logic.
- **`src/lead/learning/`**: Online learning subsystem, PID controller tuning, distribution drift detection, and Federated Averaging (`FedAvg`).
- **`src/lead/query/`**: Multi-node ordered range query execution and successor forwarding.
- **`src/lead/routing/`**: Chord ring routing, bootstrap joining, and periodic stabilization loops.
- **`src/rmi/`**: Two-stage Learned Index (RMI) mapping feature space to 64-bit ring hash space.
- **`src/storage.rs`**: Ordered key-value storage engine (`KeyStore`, `InMemoryStore`, `SledStore`, and `StorageEngine` for persistent disk-backed storage).
- **`src/ring.rs`**: 64-bit ring arithmetic, `PeerHASH`, base-10 finger table calculations, and interval logic.
- **`src/transport/`**: Tonic gRPC server and client implementations.
- **`src/api/`**: Axum HTTP REST API.

## Build & Test

```bash
cargo build
cargo test
cargo clippy --all-targets
```
