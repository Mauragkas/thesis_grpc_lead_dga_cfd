# Use Case: Start LEAD Node

## Description

Starts the `lead-node` binary, loads runtime configuration from environment variables, initializes the pluggable ordered storage engine (in-memory `InMemoryStore` or persistent disk-backed `SledStore` via `StorageEngine`), instantiates the `LeadNode` coordinator with $k$ virtual nodes, restores any persisted key counts into the FRM learned index, optionally joins an existing ring cluster, spawns periodic maintenance loops, and starts both the gRPC and HTTP REST servers.

## Primary actor

- System operator / deployment environment

## Supporting systems

- `KeyStore` (`StorageEngine` wrapping `InMemoryStore` or persistent `SledStore`)
- `GrpcRemote` (tonic gRPC client pool)
- `axum` HTTP server
- `tonic` gRPC server
- Peer LEAD nodes

## Preconditions

- The host network ports for HTTP and gRPC bindings are free.
- Environment variables may override default binding addresses, vnode counts, and storage backend settings.

## Main steps

1. The process initializes dual logging via `tracing_subscriber`: human-readable plain text on stdout for `docker logs`, and an optional structured JSON file layer for Fluent-Bit log forwarding when `LOG_FILE_PATH` or `LOG_DIR` is configured.
2. Runtime configuration is loaded with `Config::from_env()`.
3. The storage engine is instantiated based on `cfg.storage_backend`:
   - If `StorageBackend::Sled`: opens persistent database via `StorageEngine::sled(path)` (default path `"./data/lead"`).
   - If `StorageBackend::Memory`: allocates in-memory B-tree storage via `StorageEngine::memory()`.
4. A `GrpcRemote` client adapter is instantiated.
5. The `LeadNode` coordinator is created:
   - Derives $k$ virtual node IDs (VIDs) using `PeerHASH` on `"{i}|{self_uri}"`.
   - Initializes $k$ `VirtualNode` instances with base-10 finger tables.
   - Links initial successors locally in a circular ring.
   - Instantiates the initial `LearnedIndex` with default `RmiModel`.
6. `lead.init_from_storage().await;` is called to:
   - Inspect existing keys in storage and initialize `LearnedIndex::keys_total`.
   - Query `storage.get_meta("active_model")` to recover and activate any previously persisted `RmiModel` (preserving model version and parameters across node restarts).
7. If `join_uri` is configured:
   - An asynchronous join loop contacts the bootstrap node to discover successors for each local vnode.
   - Retries up to `join_retry_count` times (default 300) with `join_retry_delay_secs` sleeps (default 1s) until all vnodes discover remote successors.
8. The background maintenance tasks are spawned:
   - `stabilize_all()` (every 2s)
   - `fix_fingers_all()` (every 5s)
   - `check_predecessor_all()` (every 10s)
   - `heartbeat_round()` (every 8s)
   - `maybe_retrain()` (every 5s)
9. The tonic gRPC server is spawned on `cfg.grpc_bind`.
10. The axum HTTP server binds to `cfg.http_bind` and starts listening for REST requests.

## Postconditions

- The node is active, serving gRPC and HTTP requests, and maintaining its ring state in the cluster.

## Failure cases

- Port binding failure on `HTTP_BIND` or `GRPC_BIND`.
- Bootstrap node permanently unreachable (join loop logs status and times out).
