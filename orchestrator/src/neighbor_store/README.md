# Neighbor Store Module

Multi-probe nearest-neighbor store mapping continuous gene vectors to the LEAD DHT.

## Files and Code Structure

### 1. `trait.rs`
- **`NeighborStore`** (async trait):
  ```rust
  #[async_trait::async_trait]
  pub trait NeighborStore: Send + Sync {
      async fn store(&self, genes: &[f64], fitness: f64, generation: usize) -> Result<(), tonic::Status>;
      async fn query_knn(&self, query: &[f64], k: usize) -> Result<Vec<GenePayload>, tonic::Status>;
  }
  ```

---

### 2. `hilbert.rs`
- **`HilbertNeighborStore<G, L>`**:
  - **Type Bounds**: `G: GeneKeyGenerator`, `L: LeadStore`.
  - **Fields**:
    - `keygen: G`: Generates multi-curve Hilbert keys.
    - `store: L`: LEAD DHT client.
    - `metric: EuclideanDistance`: Ground-truth distance metric.
  - **Methods**:
    - `pub fn new(keygen: G, store: L) -> Self`: Constructor.
    - `pub async fn store(&self, genes: &[f64], fitness: f64, generation: usize) -> Result<(), Status>`:
      1. Constructs JSON `GenePayload`.
      2. Generates 3 rotated Hilbert keys (`keygen.keys_for(genes)`).
      3. Persists payload under all 3 keys via `store.store_gene()`.
    - `pub async fn query_knn(&self, query: &[f64], k: usize) -> Result<Vec<GenePayload>, Status>`:
      1. Generates 3 probe keys (`keygen.keys_for(query)`).
      2. Executes `store.range_query(key, k)` across all 3 curves.
      3. Deserializes and deduplicates candidates by key.
      4. Computes true Euclidean distance $d(\text{query}, \text{candidate})$ for each candidate.
      5. Sorts by ascending distance and returns top-$k$ nearest `GenePayload` records.
