# Use Case: Multi-Probe Hilbert Neighbor Store

## Description

Indexes evaluated continuous gene vectors across multiple rotated Hilbert space-filling curves and stores them in the LEAD DHT. Enables approximate k-nearest-neighbor (k-NN) queries by executing 1D range scans across all curves in the DHT and re-ranking candidates using true Euclidean distance.

## Primary actor

- `HilbertNeighborStore`

## Supporting systems

- `HilbertKeyGenerator` / `HilbertEncoder` (from the unified `hilbert_rs` crate, single source of truth for Hilbert encoding across Rust and Python)
- `LeadStore` / `GrpcLeadStore` (LEAD DHT routed storage and range queries with 5s timeouts)
- `EuclideanDistance` (metric ground truth)

## Preconditions

- `LEAD_ENDPOINT` is configured and connected.
- `genes_len` matches the expected vector dimensionality (10 genes).

## Main steps: Store Gene under Multi-Probe Keys

1. A newly evaluated individual `(genes, fitness, generation)` is passed to `store()`.
2. A `GenePayload` is constructed and JSON-serialized.
3. `HilbertKeyGenerator::keys_for(genes)` generates `NUM_CURVES` (3) distinct order-preserving keys:
   - For each curve $c \in \{0, 1, 2\}$, the gene coordinates are permuted using a deterministic axis rotation.
   - The point is discretized and encoded into a hex Hilbert scalar using the Skilling algorithm.
   - The key is formatted as `{curve_hex}{hex_hilbert}|{canonical_json}`.
4. **Concurrent Multi-Probe Store**: Storage futures for all generated keys are dispatched concurrently using `futures::future::try_join_all(store_futs)`, persisting the individual into each curve's partition in parallel.

## Main steps: Query k-Nearest Neighbors (k-NN)

1. A query point `query` and integer `k` are passed to `query_knn(query, k)`.
2. `HilbertKeyGenerator::keys_for(query)` generates the probe start keys for all 3 curves.
3. **Concurrent Multi-Probe Fan-Out**:
   - `LeadStore::range_query` requests are initiated concurrently across all 3 probe keys via `futures::future::join_all(query_futs)`.
   - Each curve queries $k$ candidates from the LEAD DHT.
   - Returned JSON payloads are deserialized and deduplicated by key into a `HashMap<String, GenePayload>` candidate map.
4. For each unique candidate:
   - The exact Euclidean distance $d(query, candidate)$ is computed via `EuclideanDistance`.
5. Candidates are sorted ascending by true Euclidean distance.
6. The top `k` closest `GenePayload` records are returned.

## Postconditions

- Evaluated points are indexed across multiple curves, overcoming 1D Hilbert boundary discontinuities.
- Nearest neighbors in high-dimensional continuous gene space are accurately retrieved from the distributed DHT.

## Failure cases

- LEAD node range query timeout or DHT node failure.
- Deserialization errors for corrupted payloads (skipped gracefully).
