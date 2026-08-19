# Use Case: Multi-Probe Hilbert Neighbor Store

## Description

Indexes evaluated continuous gene vectors across multiple rotated Hilbert space-filling curves and stores them in the LEAD DHT. Enables approximate k-nearest-neighbor (k-NN) queries by executing 1D range scans across all curves in the DHT and re-ranking candidates using true Euclidean distance.

## Primary actor

- `HilbertNeighborStore`

## Supporting systems

- `HilbertKeyGenerator` / `HilbertEncoder` (Skilling multi-curve indexing)
- `LeadStore` / `GrpcLeadStore` (LEAD DHT routed storage and range queries)
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
4. For each key, `LeadStore::store_gene(key, payload_json)` is called.

## Main steps: Query k-Nearest Neighbors (k-NN)

1. A query point `query` and integer `k` are passed to `query_knn(query, k)`.
2. `HilbertKeyGenerator::keys_for(query)` generates the probe start keys for all 3 curves.
3. For each curve probe key:
   - `LeadStore::range_query(key, k)` retrieves candidate entries along that curve's 1D order in the DHT.
   - Returned JSON payloads are deserialized and deduplicated by key into a candidate map.
4. For each unique candidate:
   - The exact Euclidean distance $d(query, candidate)$ is computed via `EuclideanDistance`.
5. Candidates are sorted by ascending Euclidean distance.
6. The top `k` closest `GenePayload` records are returned.

## Postconditions

- Evaluated points are indexed across multiple curves, overcoming 1D Hilbert boundary discontinuities.
- Nearest neighbors in high-dimensional continuous gene space are accurately retrieved from the distributed DHT.

## Failure cases

- LEAD node range query timeout or DHT node failure.
- Deserialization errors for corrupted payloads (skipped gracefully).
