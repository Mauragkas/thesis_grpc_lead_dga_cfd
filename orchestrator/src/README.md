# Orchestrator Source Code

Source modules for the `orchestrator` crate.

## Root Files

### 1. `config.rs`
Single source of truth for runtime configurations.

- **`GaConfig`**:
  - Fields: `pop_size: usize`, `genes_len: usize`, `generations: usize`, `mut_sigma: f64`, `elite_frac: f64`, `batch_size: usize`, `seed: u64`, `eval_endpoint: String`.
- **`TierConfig`**:
  - Fields: `epsilon_exact: f64` (default `0.005`), `radius_r: f64` (default `0.15`), `k_neighbors: usize` (default `15`), `min_neighbors: usize` (default `1`).
- **`SurrogateClientConfig`**:
  - Fields: `endpoint: Option<String>` (env `SURROGATE_ENDPOINT`).
- **`TransportConfig`**:
  - Fields: `rpc_timeout`, `max_attempts`, `retry_delay`, `channel_ready_deadline`, `connect_timeout`, `request_timeout`, `keep_alive_timeout`, `tcp_keepalive`.
- **`GeneStoreConfig`**:
  - Fields: `max_age_generations: usize` (default: `5`).
- **`LeadConfig`**:
  - Fields: `endpoint: Option<String>`.
- **`config_from_env()`**:
  - Reads all environment variable overrides.
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
  - `encode_hex(&self, point: &[f64], curve: usize, hex_width: usize) -> String`: Normalizes, scales, and encodes a vector to a zero-padded hex Hilbert scalar.
- **`HilbertKeyGenerator`**:
  - Implements `GeneKeyGenerator` producing keys formatted as `{curve_hex}{hex_hilbert}|{canonical_json}`.

---

### 3. `proto.rs`
Contains tonic-generated protobuf bindings:
- `eval`: `evaluator_client::EvaluatorClient`, `BatchRequest`, `BatchResponse`, `Individual`.
- `lead`: `lead_client::LeadClient`, `PutRoutedRequest`, `KeyMsg`, `RangeRequest`, `RangeResponse`.
- `ring`: `ring_server::Ring`, `ring_client::RingClient`, `FindSuccRequest`, `NotifyRequest`, `MigrateRequest`, `MigrateResponse`.
- `surrogate`: `surrogate_service_client::SurrogateServiceClient`, `BatchPredictRequest`, `BatchPredictResponse`, `IngestSamplesRequest`.

---

## Subsystem Directories

- [`evaluator/`](evaluator/README.md): Multi-Tier ($\epsilon$-Bypass) evaluation pipeline, tier metrics, and worker simulation client.
- [`surrogate_client/`](surrogate_client/README.md): gRPC client adapter for the external `surrogate_node` microservice.
- [`ga/`](ga/README.md): Genetic algorithm runner and pure population operators.
- [`gene_store/`](gene_store/README.md): Local exact cache and generation-age eviction.
- [`lead_store/`](lead_store/README.md): Client adapter for LEAD DHT persistence and range queries.
- [`migration/`](migration/README.md): Island-model migration, top-K selection, and migrant buffer.
- [`neighbor_store/`](neighbor_store/README.md): Multi-probe Hilbert nearest-neighbor store.
- [`ring/`](ring/README.md): Chord-like stabilizing ring topology and gRPC server.
- [`transport/`](transport/README.md): Channel construction and readiness polling.
