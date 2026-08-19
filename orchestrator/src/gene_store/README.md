# Gene Store Module

Local in-memory exact gene cache and generation-age eviction policies.

## Files and Code Structure

### 1. `trait.rs`
- **`GeneStore`** (async trait):
  ```rust
  #[async_trait::async_trait]
  pub trait GeneStore: Send + Sync {
      async fn store(&self, genes: Vec<f64>, fitness: f64, generation: usize);
      async fn query_knn(&self, query: &[f64], k: usize, current_generation: usize) -> Vec<GeneRecord>;
      async fn lookup_exact(&self, genes: &[f64], current_generation: usize) -> Option<f64>;
      async fn evict_expired(&self, current_generation: usize);
  }
  ```

---

### 2. `record.rs`
- **`GeneRecord`** (struct):
  - `pub genes: Vec<f64>`: Stored gene vector.
  - `pub fitness: f64`: Evaluated fitness score.
  - `pub generation: usize`: Generation index when evaluated.
  - `pub last_accessed_gen: usize`: Most recent generation index when retrieved (updated on cache hits).

---

### 3. `metric.rs`
- **`DistanceMetric`** (trait):
  - `fn distance(&self, a: &[f64], b: &[f64]) -> f64`
- **`EuclideanDistance`** (struct):
  - Implements $L_2$ Euclidean distance: $d(a, b) = \sqrt{\sum (a_i - b_i)^2}$.

---

### 4. `eviction.rs`
- **`EvictionPolicy`** (trait):
  - `fn should_evict(&self, record: &GeneRecord, current_generation: usize) -> bool`
- **`GenerationEvictor`** (struct):
  - `pub max_age: usize`: TTL threshold.
  - Eviction condition: `current_generation.saturating_sub(record.last_accessed_gen) > self.max_age`.

---

### 5. `in_memory.rs`
- **`InMemoryGeneStore<M: DistanceMetric, E: EvictionPolicy>`**:
  - `records: tokio::sync::Mutex<Vec<GeneRecord>>`
  - `metric: M`
  - `eviction: E`
  - `pub fn new(metric: M, eviction: E) -> Self`: Constructor.
  - `lookup_exact(&self, genes: &[f64], current_generation: usize) -> Option<f64>`: Linear scan checking bit-exact equality; updates `last_accessed_gen` on hit.
  - `store(&self, genes: Vec<f64>, fitness: f64, generation: usize)`: Appends a new `GeneRecord`.
  - `query_knn(&self, query: &[f64], k: usize, current_generation: usize) -> Vec<GeneRecord>`: Computes distances to all records and returns top-$k$ nearest.
  - `evict_expired(&self, current_generation: usize)`: Retains only records satisfying `!eviction.should_evict(record, current_generation)`.
