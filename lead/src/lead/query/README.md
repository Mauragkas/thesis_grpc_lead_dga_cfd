# Query Subsystem Module

Handles distributed range queries, overscanned scans, candidate filtering, and recursive multi-node forwarding.

## Files and Code Structure

### 1. `handle.rs`
- **`LeadNode` Range Query Methods**:
  - `pub async fn handle_range_query(&self, start_key: &str, count: u64, caller: &str, model_version: u64) -> RangeResult`:
    1. Uses active/pinned `RmiModel` to predict start hash ID: $id = \text{model.predict}(start\_key)$.
    2. Identifies local owning vnode via `find_owning_vnode(id)`.
    3. Performs local overscanned range scan in `storage` ($\text{overscan} = 3 \cdot \text{count}$).
    4. Filters keys with `collect_owned()` (retaining only keys owned under the model).
    5. Calls `complete_or_forward()`.
  - `pub async fn handle_range_forward(&self, from_key: &str, count: u64, caller: &str, origin_vid: u64, model_version: u64, mut payload: Vec<(String, String)>) -> RangeResult`:
    1. Identifies the local vnode immediately following `from_key`.
    2. Performs `range_scan_after(from_key, overscan)`.
    3. Appends filtered owned keys to `payload`.
    4. Calls `complete_or_forward()`.
  - `async fn collect_owned(&self, local: Vec<(String, String)>, model: &RmiModel, need: usize) -> Vec<(String, String)>`:
    Filters scanned keys checking `self.owns_key_with_model(k, model)`.
  - `async fn complete_or_forward(&self, payload, need, start_key, vnode, origin_vid, caller, model_version) -> RangeResult`:
    If `payload.len() >= need` or successor wraps to `origin_vid`: returns `RangeResult { entries, complete: true }`. Otherwise forwards via `remote.range_forward()`.

---

### 2. `model.rs`
- **`LeadNode` Model Helpers**:
  - `pub async fn get_model_for_version(&self, version: u64) -> RmiModel`: Returns pinned version model or active model.
  - `pub async fn owns_key(&self, key: &str) -> bool`: Checks if the key maps to any hosted vnode.
  - `pub async fn owns_key_with_model(&self, key: &str, model: &RmiModel) -> bool`: Checks key ownership against a specific model.
  - `pub async fn lookup_target(&self, key: &str) -> NodeAddr`: Finds the responsible node address for `key`.
