# LEAD DHT Node Crate

The `lead-node` crate implements the LEAD Chord-like Distributed Hash Table with online learning, Two-Stage Recursive Model Indexing (RMI), online 2-bit PID tuning, virtual nodes, multi-probe Hilbert key normalization, durable Sled storage with active model recovery, and Federated Averaging (`FedAvg`).

## Architecture

- **`src/lead/`**: Physical `LeadNode` coordinator, `VirtualNode` vnode management, and routing logic.
- **`src/lead/learning/`**: Online learning subsystem (`LearnedIndex`), PID controller tuning, distribution drift detection, and Federated Averaging (`FedAvg`).
- **`src/lead/query/`**: Multi-node ordered range query execution with parallel scatter-gather fanout and peer deduplication.
- **`src/lead/routing/`**: Chord ring routing, bootstrap joining, key migration, and periodic stabilization loops.
- **`src/rmi/`**: Two-stage Learned Index (RMI) mapping continuous feature space to 64-bit ring hash space with normalized multi-probe curves.
- **`src/storage/`**: Pluggable ordered key-value storage engine (`KeyStore`, `InMemoryStore`, `SledStore`, and `StorageEngine` strategy wrapper) supporting metadata persistence for model version survival.
- **`src/ring.rs`**: 64-bit ring arithmetic, `PeerHASH`, base-10 finger table calculations, and interval logic.
- **`src/transport/`**: Tonic gRPC server and client implementations.
- **`src/api/`**: Axum HTTP REST API for key-value CRUD and cluster health inspection.

## Key Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `LEAD_STORAGE_BACKEND` / `STORAGE_BACKEND` | `memory` | Storage backend (`memory` or `sled`) |
| `LEAD_STORAGE_PATH` / `STORAGE_PATH` | `./data/lead` | Directory path for persistent sled database |
| `HTTP_BIND` | `0.0.0.0:8080` | Bind address for Axum HTTP REST server |
| `GRPC_BIND` | `0.0.0.0:50051` | Bind address for Tonic gRPC server |
| `SELF_URI` | `http://127.0.0.1:50051` | Public gRPC address advertised to peers |
| `JOIN_URI` | `None` | Optional seed peer address to join ring |
| `VIRTUAL_NODE_COUNT` | `10` | Number of virtual nodes per physical instance |
| `LEAD_SUCCESSOR_LIST_LEN` | `4` | Successor list length for ring fault-tolerance |
| `LEAD_NUM_CURVES` / `NUM_CURVES` | `16` | Number of multi-probe curves for feature normalization |
| `LEAD_RANGE_OVERSCAN_MULTIPLIER` | `3` | Multiplier for local range query overscan |
| `LEAD_DRIFT_THRESHOLD` | `0.40` | Drift fraction triggering model retraining |
| `LEAD_MIN_KEYS_FOR_DRIFT` | `50` | Minimum keys before drift evaluation begins |
| `LEAD_FRM_QUORUM_THRESHOLD` | `0.90` | Quorum ratio for Transient Coordinator election |
| `LOG_FILE_PATH` / `LOG_DIR` | `None` | Path / directory for structured JSON file logging |

## Build & Test

```bash
# Build binary and library
cargo build --release

# Run unit and integration tests
cargo test

# Check code formatting and lints
cargo clippy --all-targets
```

For detailed architectural specifications, domain models, sequence diagrams, and use cases, see [`docs/README.md`](docs/README.md).
