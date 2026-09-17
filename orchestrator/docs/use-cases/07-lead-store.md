# Use Case: Persist and Query Genes in LEAD DHT

## Description

Stores and retrieves serialized gene payloads and executes lexicographic range queries against a distributed LEAD Chord DHT node using the routed gRPC interface.

## Primary actor

- `GrpcLeadStore` / `HilbertNeighborStore`

## Supporting systems

- `LeadStore` abstraction
- `LeadClient` gRPC service (`/lead.Lead/PutRouted`, `/lead.Lead/GetRouted`, `/lead.Lead/RangeQuery`)

## Preconditions

- `LEAD_ENDPOINT` is configured.
- The `LeadClient` channel is established.

## Main steps: Store Gene

1. A JSON-serialized `GenePayload` and key are supplied.
2. `store_gene(key, value)` is invoked on `GrpcLeadStore`.
3. `LeadClient::put_routed` is invoked with `PutRoutedRequest { key, value }`, wrapped in a 5-second timeout (`tokio::time::timeout`).
4. The LEAD node routes the request to the responsible vnode on the DHT ring and stores it.

## Main steps: Get Gene

1. A key is supplied.
2. `get_gene(key)` calls `LeadClient::get_routed` with `KeyMsg { key }`, wrapped in a 5-second timeout.
3. The responsible node returns the stored JSON payload, or `NotFound` (returning `Ok(None)`).

## Main steps: Range Query

1. A `start_key` and integer `count` limit are supplied.
2. `range_query(start_key, count)` calls `LeadClient::range_query` with `RangeRequest { start_key, count, caller_address }`, wrapped in a 5-second timeout.
3. The LEAD cluster executes a parallel scatter-gather fan-out across known peers and returns a globally sorted list of matching `(key, value)` entries in Hilbert order.

## Postconditions

- Genes are durably stored in the distributed DHT and can be retrieved or range-scanned for nearest neighbor discovery.
- All network operations are protected by explicit 5-second timeouts, preventing hang states.

## Failure cases

- DHT node unavailable or connection reset.
- gRPC deadline exceeded (5s timeout fires, returning `Status::deadline_exceeded`).
- Storage/routing failure inside the LEAD cluster.

