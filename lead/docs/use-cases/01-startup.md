# Use Case: Start LEAD Node

## Description

Starts the `lead-node` binary, loads runtime configuration from environment variables, initializes the in-memory ordered storage engine, instantiates the `LeadNode` coordinator with $k$ virtual nodes, optionally joins an existing ring cluster, spawns periodic maintenance loops, and starts both the gRPC and HTTP REST servers.

## Primary actor

- System operator / deployment environment

## Supporting systems

- `InMemoryStore` (ordered BTreeMap storage)
- `GrpcRemote` (tonic gRPC client pool)
- `axum` HTTP server
- `tonic` gRPC server
- Peer LEAD nodes

## Preconditions

- The host network ports for HTTP and gRPC bindings are free.
- Environment variables may override default binding addresses and vnode counts.

## Main steps

1. The process initializes structured JSON logging via `tracing_subscriber`.
2. Runtime configuration is loaded with `Config::from_env()`.
3. An `InMemoryStore` and a `GrpcRemote` client adapter are instantiated.
4. The `LeadNode` coordinator is created:
   - Derives $k$ virtual node IDs (VIDs) using `PeerHASH` on `"{i}|{self_uri}"`.
   - Initializes $k$ `VirtualNode` instances with base-10 finger tables.
   - Links initial successors locally in a circular ring.
   - Instantiates the initial `LearnedIndex` with default `RmiModel`.
5. If `join_uri` is configured:
   - An asynchronous join loop contacts the bootstrap node to discover successors for each local vnode.
   - Retries up to 300 times (with 1-second sleeps) until all vnodes discover remote successors.
6. The background maintenance tasks are spawned:
   - `stabilize_all()` (every 2s)
   - `fix_fingers_all()` (every 5s)
   - `check_predecessor_all()` (every 10s)
   - `heartbeat_round()` (every 8s)
   - `maybe_retrain()` (every 5s)
7. The tonic gRPC server is spawned on `cfg.grpc_bind`.
8. The axum HTTP server binds to `cfg.http_bind` and starts listening for REST requests.

## Postconditions

- The node is active, serving gRPC and HTTP requests, and maintaining its ring state in the cluster.

## Failure cases

- Port binding failure on `HTTP_BIND` or `GRPC_BIND`.
- Bootstrap node permanently unreachable (join loop logs status and times out).
