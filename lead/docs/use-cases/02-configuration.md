# Use Case: Load Configuration from Environment

## Description

Reads environment variables to configure network interfaces, cluster bootstrap endpoints, public advertised addresses, and virtual node counts.

## Primary actor

- Deployment environment / Node startup

## Environment variables

- `HTTP_BIND`: Local bind socket address for the axum HTTP REST API (default: `"0.0.0.0:8080"`).
- `GRPC_BIND`: Local bind socket address for the tonic gRPC server (default: `"0.0.0.0:50051"`).
- `SELF_URI`: Advertised gRPC endpoint URI used by peer nodes to reach this node (default: `"http://127.0.0.1:50051"`).
- `JOIN_URI`: Optional gRPC endpoint URI of an existing cluster member to join.
- `VIRTUAL_NODE_COUNT`: Number of virtual nodes ($k$) hosted by this physical instance (default: `10`).

## Main steps

1. `Config::from_env()` is called during startup.
2. Read `HTTP_BIND` or fallback to `"0.0.0.0:8080"`.
3. Read `GRPC_BIND` or fallback to `"0.0.0.0:50051"`.
4. Read `SELF_URI` or fallback to `"http://127.0.0.1:50051"`.
5. Read optional `JOIN_URI`.
6. Parse `VIRTUAL_NODE_COUNT` as integer or fallback to `10`.
7. Return initialized `Config` struct.

## Postconditions

- A valid `Config` object is provided to `LeadNode` and server builders.
