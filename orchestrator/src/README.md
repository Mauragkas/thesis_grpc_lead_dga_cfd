# Orchestrator Source Code

Source modules for the `orchestrator` crate.

## Root Files

### 1. `config.rs`
Single source of truth for runtime configurations.

- **`GaConfig`**:
  - Fields: `pop_size: usize`, `genes_len: usize`, `generations: usize`, `mut_sigma: f64`, `elite_frac: f64`, `batch_size: usize`, `seed: u64`, `eval_endpoint: String`.
  - Default: `pop_size=60`, `genes_len=10`, `generations=10`, `mut_sigma=0.08`, `elite_frac=0.5`, `batch_size=1`, `seed=42`, `eval_endpoint="load-balancer:50051"`.
- **`TransportConfig`**:
  - Fields: `rpc_timeout`, `max_attempts`, `retry_delay`, `channel_ready_deadline`, `connect_timeout`, `request_timeout`, `keep_alive_timeout`, `tcp_keepalive`.
- **`GeneStoreConfig`**:
  - Fields: `max_age_generations: usize` (default: `5`).
- **`LeadConfig`**:
  - Fields: `endpoint: Option<String>`.
- **`config_from_env() -> (GaConfig, TransportConfig, GeneStoreConfig, LeadConfig, RingConfig, MigrationConfig)`**:
  - Reads `EVAL_ENDPOINT`, `GA_SEED`, `GENE_STORE_MAX_AGE`, `LEAD_ENDPOINT`, `RING_BIND`, `RING_SELF_ADDRESS`, `RING_BOOTSTRAP`, `MIGRATION_INTERVAL`, `MIGRATION_COUNT`.
- **`clip(x: f64) -> f64`**:
  - Clamps a floating-point gene value to $[0.0, 1.0]$.

---

### 2. `hilbert.rs`
Multi-probe Compact Hilbert curve embedding (Skilling 2004 algorithm).

- **Constants**:
  - `BITS = 16`: Bit precision per dimension ($10 \times 16 = 160\text{ bits}$).
  - `NUM_CURVES = 3`: Number of distinct rotated coordinate permutations.
  - `PERM_SEED = 20240807`: Deterministic seed matching Python reference.
- **`HilbertEncoder`**:
  - `new(ndims: usize, bits: usize) -> Self`: Initializes coordinate permutation tables.
  - `encode_hex(&self, point: &[f64], curve: usize, hex_width: usize) -> String`: Normalizes, scales, and encodes a vector to a zero-padded hex Hilbert scalar.
- **`GeneKeyGenerator`** (Trait):
  - `fn keys_for(&self, genes: &[f64]) -> Vec<String>`
  - `fn parse_key(&self, key: &str) -> Option<Vec<f64>>`
- **`HilbertKeyGenerator`**:
  - Implements `GeneKeyGenerator` producing keys formatted as `{curve_hex}{hex_hilbert}|{canonical_json}`.

---

### 3. `proto.rs`
Contains tonic-generated protobuf bindings:
- `eval`: `evaluator_client::EvaluatorClient`, `BatchRequest`, `BatchResponse`, `Individual`.
- `lead`: `lead_client::LeadClient`, `PutRoutedRequest`, `KeyMsg`, `RangeRequest`, `RangeResponse`.
- `ring`: `ring_server::Ring`, `ring_client::RingClient`, `FindSuccRequest`, `NotifyRequest`, `MigrateRequest`, `MigrateResponse`.

---

## Subsystem Directories

- [`evaluator/`](evaluator/README.md): gRPC worker evaluation client.
- [`ga/`](ga/README.md): Genetic algorithm runner and pure population operators.
- [`gene_store/`](gene_store/README.md): Local exact cache and generation-age eviction.
- [`lead_store/`](lead_store/README.md): Client adapter for LEAD DHT persistence and range queries.
- [`migration/`](migration/README.md): Island-model migration, top-K selection, and migrant buffer.
- [`neighbor_store/`](neighbor_store/README.md): Multi-probe Hilbert nearest-neighbor store.
- [`ring/`](ring/README.md): Chord-like stabilizing ring topology and gRPC server.
- [`transport/`](transport/README.md): Channel construction and readiness polling.
