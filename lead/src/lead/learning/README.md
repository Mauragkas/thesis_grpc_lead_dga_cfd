# Learning Subsystem Module

Online learning, distribution drift tracking, PID controller tuning, and Federated Averaging (`FedAvg`).

## Files and Code Structure

### 1. `index.rs`
- **`LearnedIndex`** (struct):
  - **Methods**:
    - `pub fn new() -> Self`: Initializes with default `RmiModel` (version 1) and grace period tracker.
    - `pub async fn predict(&self, key: &str) -> NodeId`: Predicts 64-bit LearnedHASH.
    - `pub async fn record_insert(&self, key: &str, grace_secs, min_keys, threshold) -> bool`: Tracks newly inserted keys. If drift ratio $\text{drift\_new} / \text{keys\_total} \ge \text{threshold}$, sets `update_ready = true`. Returns `true` if PID interval is reached.
    - `pub async fn active_model(&self) -> RmiModel`: Returns currently active model.
    - `pub async fn set_pending_model(&self, model: RmiModel)`: Sets candidate model.
    - `pub async fn activate(&self, model: RmiModel)`: Activates candidate model and clears pending state.
    - `pub async fn is_update_ready(&self) -> bool`: Returns `update_ready` status.
    - `pub async fn accept_pushed_model(&self, version: u64, data: &[u8]) -> bool`: Validates and accepts higher-version model pushed by coordinator.
    - `pub async fn reset_drift(&self, keys_total: usize)`: Resets `drift_new = 0` and `update_ready = false`.

---

### 2. `pid.rs`
- **`PidTuner`** (struct):
  - **Constants / Fields**: `target_ratio=0.95`, `scale_step=0.05`, `centering_step=0.01`, `upper_threshold=0.05`, `mid_threshold=0.02`, `min_samples=20`.
  - **Methods**:
    - `pub fn adjust(&self, state: &mut PidState, anchor: &mut Anchor, correct: usize, outside: usize)`:
      Computes lookup accuracy ratio. If error $\ne 0$, adjusts `anchor.scale` and `anchor.offset`, updating proportional, integral, and derivative state. Clamps scale to $[0.5, 2.0]$.

---

### 3. `fed_avg.rs`
- **`pub(super) fn fed_avg(models: Vec<RmiModel>, version: u64) -> RmiModel`**:
  - Aggregates collected models from neighbor nodes into a single unified `RmiModel`:
    $$w_{\text{global}} = \sum \frac{n_i}{N_{\text{total}}} w_i, \quad b_{\text{global}} = \sum \frac{n_i}{N_{\text{total}}} b_i, \quad \text{offset}_{\text{global}} = \sum \frac{n_i}{N_{\text{total}}} \text{offset}_i$$

---

### 4. `training.rs`
- **`LeadNode` methods**:
  - `pub(super) async fn run_global_training_round(&self)`: Transient Coordinator workflow: gathers neighbor models (`collect_neighbor_models`), computes `fed_avg`, broadcasts global model (`broadcast_global_model`), activates model locally, and migrates keys.
  - `pub(super) async fn migrate_keys_for_new_model(&self)`: Iterates local keys; if a key is no longer owned under the new model, sends `put_local` to the new owner and removes it locally.
  - `pub(super) async fn reset_drift_state(&self)`: Resets drift counters.

---

### 5. `prune.rs`
- **`LeadNode` methods**:
  - `prune_low_throughput_vnodes()`: Identifies vnodes with elevated error rates or zero throughput and initiates graceful key handover before pruning.
