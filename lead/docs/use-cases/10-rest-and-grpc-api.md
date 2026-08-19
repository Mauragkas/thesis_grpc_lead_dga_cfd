# Use Case: Serve gRPC and HTTP REST APIs

## Description

Exposes dual communication interfaces: high-throughput tonic gRPC for inter-node clustering and high-performance client operations, and an axum HTTP REST API for external tooling, inspection, and health monitoring.

## Primary actor

- External clients (orchestrators, benchmarks, web browsers, monitoring dashboards)
- Peer LEAD nodes

## Supporting systems

- `tonic` gRPC server (binds `GRPC_BIND`)
- `axum` HTTP server (binds `HTTP_BIND`)
- `LeadNode` coordinator

## gRPC Interface (`lead.proto`)

| RPC | Description |
| :--- | :--- |
| `FindSuccessor` | Resolves the successor vnode for a target 64-bit identifier. |
| `GetPredecessor` | Retrieves the immediate predecessor of a local vnode. |
| `Notify` | Informs a local vnode of a candidate predecessor. |
| `GetSuccessor` / `GetSuccessorList` | Retrieves immediate successor or backup successor list. |
| `GetLocal` / `PutLocal` / `DeleteLocal` | Direct non-routed local storage operations. |
| `GetRouted` / `PutRouted` | Routed storage operations using LearnedHASH prediction and Chord lookup. |
| `RangeQuery` / `RangeForward` | Multi-node ordered range query scanning and forwarding. |
| `PushModel` / `RequestModel` | Model synchronization and federated learning distribution. |
| `Heartbeat` | Peer health checks and model update readiness status. |
| `GetKeys` | Returns snapshot list of local storage keys. |
| `Ping` | Node liveness health verification. |

## HTTP REST Interface (axum)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Health check endpoint returning status `"ok"`. |
| `/successor` | `GET` | Returns primary successor address of the first local vnode. |
| `/predecessor` | `GET` | Returns predecessor node info. |
| `/find_successor/:id` | `GET` | Resolves successor for integer hash `:id`. |
| `/notify` | `POST` | Submits predecessor notification payload. |
| `/kv/local/:key` | `GET`, `POST`, `DELETE` | Accesses local node key-value storage directly. |
| `/kv/:key` | `GET`, `POST`, `DELETE` | Executes routed key-value operation using LearnedHASH. |
| `/keys` | `GET` | Returns JSON array of all stored keys on this node. |
| `/range?start=&count=` | `GET` | Runs distributed range query starting at `start` key. |

## Postconditions

- Both gRPC clients and HTTP clients can seamlessly query and mutate cluster state.
