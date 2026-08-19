# HTTP REST API Module

Axum HTTP REST API interface for external inspection, health checking, and key-value storage operations.

## Files and Code Structure

### 1. `mod.rs`
- **`pub type AppState<S, R> = Arc<LeadNode<S, R>>`**: Shared state injected into axum router.
- **`pub fn router<S, R>(state: AppState<S, R>) -> Router`**:
  Registers REST routes:
  - `GET /health` $\to$ Returns `"ok"`.
  - `GET /successor` $\to$ `ring::get_successor`
  - `GET /predecessor` $\to$ `ring::get_predecessor`
  - `GET /find_successor/:id` $\to$ `ring::find_successor`
  - `POST /notify` $\to$ `ring::notify`
  - `GET`, `POST`, `DELETE /kv/local/:key` $\to$ `kv::get_local_kv`, `kv::put_local_kv`, `kv::del_local_kv`
  - `GET`, `POST`, `DELETE /kv/:key` $\to$ `kv::get_kv`, `kv::put_kv`, `kv::del_kv`
  - `GET /keys` $\to$ `range::get_all_keys`
  - `GET /range` $\to$ `range::range_query`

---

### 2. `kv.rs`
- **`pub async fn get_local_kv(State(state), Path(key))`**: Reads directly from `state.storage().get(&key)`.
- **`pub async fn put_local_kv(State(state), Path(key), body)`**: Writes directly to `state.storage().put(key, body)` and records insertion in `LearnedIndex`.
- **`pub async fn del_local_kv(State(state), Path(key))`**: Removes from `state.storage().remove(&key)`.
- **`pub async fn get_kv(State(state), Path(key))`**: Predicts LearnedHASH, routes to responsible node, and returns value.
- **`pub async fn put_kv(State(state), Path(key), body)`**: Predicts LearnedHASH, routes to responsible node, and stores value.
- **`pub async fn del_kv(State(state), Path(key))`**: Routes deletion to responsible node.

---

### 3. `range.rs`
- **`pub async fn get_all_keys(State(state))`**: Returns JSON array of all stored keys on the node.
- **`pub async fn range_query(State(state), Query(params))`**: Unpacks `start` and `count` parameters and calls `state.handle_range_query(&start, count, &state.self_uri, 0)`.

---

### 4. `ring.rs`
- **`pub async fn get_successor(State(state))`**: Returns the primary successor of `vnodes[0]`.
- **`pub async fn get_predecessor(State(state))`**: Returns predecessor node info.
- **`pub async fn find_successor(State(state), Path(id))`**: Calls `state.find_successor(vnodes[0].vid, id)`.
- **`pub async fn notify(State(state), Json(other))`**: Calls `vnodes[0].notify(other)`.
