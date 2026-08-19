# Common Test Utilities

Shared test doubles and configuration helpers for orchestrator integration tests.

## Code Structure (`mod.rs`)

### 1. `MockEvaluator`
- **Fields**:
  - `pub fitnesses: Mutex<Vec<f64>>`: Pre-configured fitness returns.
  - `pub calls: AtomicUsize`: Call invocation counter.
  - `pub fail: bool`: Simulates gRPC internal errors when `true`.
- **Methods**:
  - `pub fn new(fitnesses: Vec<f64>) -> Self`: Constructor.
  - `pub fn failing() -> Self`: Constructs a failing evaluator.
  - `pub fn call_count(&self) -> usize`: Returns number of `evaluate_population` calls.
- **Trait Impl**: Implements `Evaluator` trait.

---

### 2. `MockGeneStore`
- **Fields**:
  - `pub exact: Mutex<HashMap<String, f64>>`: Pre-seeded exact cache hits.
  - `pub store_calls: AtomicUsize`: Tracks `store` invocations.
  - `pub evict_calls: AtomicUsize`: Tracks `evict_expired` invocations.
  - `pub knn_calls: AtomicUsize`: Tracks `query_knn` invocations.
- **Methods**:
  - `pub fn empty() -> Self`: Constructs empty store.
  - `pub fn with_exact(genes: &[f64], fitness: f64) -> Self`: Constructs store with pre-seeded cache hit.
  - `pub fn seed_exact(&self, genes: &[f64], fitness: f64)`: Adds exact match key.
- **Trait Impl**: Implements `GeneStore` trait.

---

### 3. `NoopMigration`
- **Trait Impl**: Implements `MigrationHook` trait with no-op `maybe_emigrate` and empty `drain_immigrants`.

---

### 4. `small_config() -> GaConfig`
- Helper creating a minimal deterministic config: `pop_size=6`, `genes_len=4`, `generations=3`, `mut_sigma=0.05`, `elite_frac=0.5`, `batch_size=2`, `seed=7`.
