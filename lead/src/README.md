# LEAD Source Code

Source modules for the `lead-node` crate.

## Root Files

### 1. `config.rs`
- **`Config`** (struct):
  - `pub http_bind: String`: Socket address for the axum HTTP REST API (default: `"0.0.0.0:8080"`).
  - `pub grpc_bind: String`: Socket address for the tonic gRPC server (default: `"0.0.0.0:50051"`).
  - `pub self_uri: String`: Public URI advertised to cluster peers (default: `"http://127.0.0.1:50051"`).
  - `pub join_uri: Option<String>`: Optional bootstrap peer URI.
  - `pub virtual_node_count: usize`: Number of virtual nodes ($k$) hosted per physical node (default: `10`).
  - `pub fn from_env() -> Self`: Parses environment variables (`HTTP_BIND`, `GRPC_BIND`, `SELF_URI`, `JOIN_URI`, `VIRTUAL_NODE_COUNT`).

---

### 2. `ring.rs`
64-bit ring arithmetic, node identity hashing, and interval operations.

- **Constants**:
  - `M: usize = 64`: Ring bit width ($2^{64}$ identifier space).
  - `F: usize = 18`: Base-10 finger table capacity ($10^0$ to $10^{17}$).
- **`NodeId`**: `type NodeId = u64`.
- **`NodeAddr`** (struct):
  - `pub id: NodeId`: 64-bit virtual node ID.
  - `pub address: String`: Socket address / URI.
- **`pub fn peer_hash(val: &str) -> NodeId`**:
  - Computes SHA-1 hash of `val` and returns first 8 bytes as big-endian `u64`. Used strictly for virtual node identifiers (VIDs), never for data keys.
- **`pub fn finger_start(id: NodeId, i: usize) -> NodeId`**:
  - Computes target key for finger entry $i$: $\text{id} + 10^{i-1}$ using wrapping addition.
- **`pub fn in_range(val: NodeId, start: NodeId, end: NodeId, inclusive_end: bool) -> bool`**:
  - Circular interval arithmetic checking if `val` lies in $(start, end]$ (if `inclusive_end=true`) or $(start, end)$ (if `false`), correctly handling wrap-around at $2^{64}-1 \to 0$.

---

### 3. `storage.rs`
Ordered in-memory storage engine.

- **`KeyStore`** (async trait):
  - `get(&self, key: &str) -> Option<String>`
  - `put(&self, key: String, val: String)`
  - `remove(&self, key: &str) -> Option<String>`
  - `range_scan(&self, start_key: &str, count: usize) -> Vec<(String, String)>`
  - `range_scan_after(&self, after_key: &str, count: usize) -> Vec<(String, String)>`
  - `snapshot(&self) -> Vec<(String, String)>`
  - `len(&self) -> usize`
  - `is_empty(&self) -> bool`
- **`InMemoryStore`** (struct):
  - Implements `KeyStore` backed by `tokio::sync::RwLock<std::collections::BTreeMap<String, String>>`.

---

## Subsystem Directories

- [`api/`](api/README.md): Axum HTTP REST endpoints.
- [`lead/`](lead/README.md): Physical `LeadNode` coordinator, `VirtualNode` vnodes, query routing, and online learning.
- [`rmi/`](rmi/README.md): Two-Stage Learned Index (RMI) and feature extraction.
- [`transport/`](transport/README.md): Client segregation traits and gRPC implementations.
