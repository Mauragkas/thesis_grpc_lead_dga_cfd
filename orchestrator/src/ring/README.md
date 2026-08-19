# Ring Subsystem Module

Chord-like self-stabilizing ring topology connecting orchestrator island nodes.

## Files and Code Structure

### 1. `state.rs`
- **`NodeInfo`** (struct):
  - `pub id: u64`: Hash ID on the ring.
  - `pub address: String`: Socket address (e.g. `"orchestrator:50052"`).
- **`RingState`** (struct):
  - `pub self_node: NodeInfo`: Identity of this node.
  - `pub predecessor: Mutex<Option<NodeInfo>>`: Immediate predecessor.
  - `pub successor: Mutex<Option<NodeInfo>>`: Immediate successor.
  - `pub successor_list: Mutex<Vec<NodeInfo>>`: Backup successor list for fault tolerance.
  - `pub fn in_half_open(a: u64, b: u64, id: u64) -> bool`: Checks if `id` $\in (a, b]$ on the circular ring.
  - `pub fn in_open(a: u64, b: u64, id: u64) -> bool`: Checks if `id` $\in (a, b)$ on the circular ring.

---

### 2. `hash.rs`
- **`AddressHasher`** (trait): `fn hash(&self, addr: &str) -> u64`.
- **`Sha256Hasher`** (struct): Computes SHA-256 hash and takes first 8 bytes as big-endian `u64`.

---

### 3. `member.rs`
- **`LocalRingMember`**:
  - `pub async fn find_successor(&self, id: u64) -> NodeInfo`: Resolves successor from local perspective.
  - `pub async fn get_predecessor(&self) -> Option<NodeInfo>`: Returns current predecessor.
  - `pub async fn notify(&self, other: NodeInfo) -> bool`: Accepts `other` if it is closer than current predecessor.
  - `pub async fn get_successor_list(&self) -> Vec<NodeInfo>`: Returns local successor list.
  - `pub async fn set_successor(&self, node: NodeInfo)`: Updates immediate successor.
  - `pub async fn set_successor_list(&self, list: Vec<NodeInfo>)`: Updates backup list.

---

### 4. `client.rs` & `grpc_client.rs`
- **`RingClient`** (async trait):
  - `find_successor`, `get_predecessor`, `notify`, `get_successor_list`, `ping`.
- **`GrpcRingClient`**:
  - Implements `RingClient` using a thread-safe `Mutex<HashMap<String, Channel>>` connection pool.

---

### 5. `grpc_server.rs`
- **`RingServer`**:
  - Implements `orchestrator_ring.Ring` tonic service:
    - `find_successor`: Delegates to `member.find_successor()`.
    - `get_predecessor`: Delegates to `member.get_predecessor()`.
    - `notify`: Delegates to `member.notify()`.
    - `get_successor_list`: Delegates to `member.get_successor_list()`.
    - `ping`: Returns `{ ok: true }`.
    - `migrate`: Pushes incoming migrants into `migrant_buffer`.

---

### 6. `stabilization.rs`
- **`Stabilizer<C: RingClient>`**:
  - `pub fn spawn(self)`: Spawns the stabilization loop in a background tokio task.
  - `pub async fn join(&self, bootstrap_addr: &str) -> Result<(), Status>`: Asks bootstrap node to find successor for `self.id`.
  - `async fn stabilize(&self) -> Result<(), Status>`: Asks successor for its predecessor $x$. If $x \in (\text{self}, \text{successor})$, updates successor to $x$ and notifies $x$; otherwise notifies current successor.
  - `async fn refresh_successor_list(&self) -> Result<(), Status>`: Fetches successor list from immediate successor and prepends it.
