# gRPC Transport Module

Tonic gRPC server and client implementation for the `lead.Lead` service.

## Subdirectories

- **`client/`**: `GrpcRemote` client adapter managing cached tonic channels and dispatching RPCs.
- **`server/`**: Tonic gRPC service handlers translating inbound protobuf requests into `LeadNode` method calls.
- **`mod.rs`**: Re-exports client and server components.
