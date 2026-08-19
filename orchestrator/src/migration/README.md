# Island Migration Module

Implements island-model genetic migration across orchestrator nodes connected on the Chord ring.

## Files and Code Structure

### 1. `config.rs`
- **`MigrationConfig`**:
  - `pub interval_generations: usize`: Generational period between migrations (default: `5`).
  - `pub migrant_count: usize`: Number of individuals to emigrate per interval (default: `3`).
- **`RingConfig`**:
  - `pub bind_address: String`: Local socket address for Ring gRPC server.
  - `pub self_address: String`: Public address advertised to ring peers.
  - `pub bootstrap_address: Option<String>`: Seed node to join the ring.
  - `pub stabilize_interval: Duration`: Stabilization loop tick (default: `2s`).
  - `pub successor_list_size: usize`: Successor list capacity (default: `4`).

---

### 2. `mod.rs`
- **`MigrantIndividual`** (struct):
  - `pub genes: Vec<f64>`: Migrating individual's gene vector.
  - `pub fitness: f64`: Fitness value.

---

### 3. `trait.rs`
- **`MigrationHook`** (async trait):
  ```rust
  #[async_trait::async_trait]
  pub trait MigrationHook: Send + Sync {
      async fn maybe_emigrate(&self, generation: usize, population: &[Vec<f64>], fitnesses: &[f64]) -> Result<(), tonic::Status>;
      async fn drain_immigrants(&self) -> Vec<MigrantIndividual>;
  }
  ```

---

### 4. `selector.rs`
- **`MigrantSelector`** (trait):
  - `fn select(&self, population: &[Vec<f64>], fitnesses: &[f64], count: usize) -> Vec<MigrantIndividual>`
- **`TopKSelector`** (struct):
  - Selects the top-$K$ individuals sorted by fitness descending.

---

### 5. `buffer.rs`
- **`MigrantBuffer`** (struct):
  - `queue: tokio::sync::Mutex<std::collections::VecDeque<MigrantIndividual>>`
  - `pub async fn push(&self, migrants: Vec<MigrantIndividual>)`: Appends incoming immigrants received from predecessor nodes.
  - `pub async fn drain(&self) -> Vec<MigrantIndividual>`: Drains all queued immigrants.

---

### 6. `lead_migration.rs`
- **`LeadMigration<S: MigrantSelector>`**:
  - **Fields**: `config: MigrationConfig`, `member: Arc<LocalRingMember>`, `selector: S`, `buffer: Arc<MigrantBuffer>`, `channels: Mutex<HashMap<String, Channel>>`.
  - **Methods**:
    - `pub fn new(config, member, selector, buffer) -> Self`: Constructor.
    - `pub async fn maybe_emigrate(&self, generation, population, fitnesses) -> Result<(), Status>`:
      If `generation % interval == 0`:
      1. Selects top-$K$ migrants via `selector.select()`.
      2. Queries ring successor from `member.state().successor`.
      3. Sends `MigrateRequest` via gRPC to successor `/orchestrator_ring.Ring/Migrate`.
    - `pub async fn drain_immigrants(&self) -> Vec<MigrantIndividual>`: Delegates to `buffer.drain()`.
