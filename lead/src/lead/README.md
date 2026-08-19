# Lead Core Module

Physical node coordinator, virtual node implementation, and Chord lookup routing.

## Files and Code Structure

### 1. `node.rs`
- **`LeadNode<S: KeyStore, R: RemoteNode>`** (struct):
  - **Fields**:
    - `pub self_uri: String`: Advertised address.
    - `pub vnodes: Vec<VirtualNode>`: Hosted virtual nodes.
    - `pub self_info: NodeAddr`: Identity of the primary vnode.
    - `pub(crate) storage: Arc<S>`: Local storage engine.
    - `pub(crate) remote: Arc<R>`: Remote gRPC client.
    - `pub(crate) learning: LearnedIndex`: Two-stage RMI and PID tuner.
    - `pub(crate) finger_fix_idx: AtomicUsize`: Round-robin finger fixing index.
  - **Methods**:
    - `pub fn new(self_uri: String, k: usize, storage: Arc<S>, remote: Arc<R>) -> Self`:
      Derives $k$ unique VIDs via `peer_hash("{i}|{self_uri}")`, constructs $k$ `VirtualNode` instances, and initializes internal circular ring pointers.
    - `pub fn storage(&self) -> Arc<S>`: Returns reference to storage.
    - `pub fn remote(&self) -> Arc<R>`: Returns reference to remote client.
    - `pub fn vnode_count(&self) -> usize`: Returns count of hosted vnodes.
    - `pub async fn vids(&self) -> Vec<NodeId>`: Returns list of hosted virtual node IDs.
    - `pub async fn learned_hash(&self, key: &str) -> NodeId`: Predicts 64-bit ring hash via `learning.predict(key)`.
    - `pub async fn is_alone(&self) -> bool`: Returns `true` if all hosted vnodes have themselves as successors.

---

### 2. `vnode.rs`
- **`VirtualNode`** (struct):
  - **Fields**:
    - `pub vid: NodeId`: Virtual node identifier on the 64-bit ring.
    - `pub predecessor: RwLock<Option<NodeAddr>>`: Immediate predecessor.
    - `pub fingers: RwLock<Vec<Option<NodeAddr>>>`: 18-entry base-10 finger table.
    - `pub successor_list: RwLock<Vec<NodeAddr>>`: Backup successor list ($R=6$).
    - `pub request_count: AtomicU64`: Number of routed requests handled.
    - `pub error_count: AtomicU64`: Number of errors encountered.
    - `pub last_active: RwLock<Instant>`: Timestamp of last request.
    - `pub pruned: AtomicU64`: Flag ($0=\text{active}, 1=\text{pruned}$).
  - **Methods**:
    - `pub async fn successor(&self) -> NodeAddr`: Returns first node in `successor_list`.
    - `pub fn error_rate(&self) -> f64`: Returns $\text{error\_count} / \text{request\_count}$.
    - `pub fn record_request(&self)`: Increments request counter.
    - `pub fn record_error(&self)`: Increments error counter.
    - `pub async fn mark_active(&self)`: Updates `last_active` timestamp.

- **`RmiState`** (struct):
  - `pub active: RmiModel`: Currently active model.
  - `pub update: Option<RmiModel>`: Pending candidate model update.
  - `pub drift_new: usize`: Count of inserted keys since last retrain.
  - `pub update_ready: bool`: Flag indicating retrain/consensus is needed.
  - `pub dirty_leaves: HashSet<usize>`: Set of modified leaf indices for differential sync.

---

### 3. `lookup.rs`
- **`pub(crate) fn in_range_ring(val: NodeId, start: NodeId, end: NodeId, inclusive_end: bool) -> bool`**: Circular interval helper.
- **`find_vnode(&self, vid: NodeId) -> Option<&VirtualNode>`**: Finds hosted vnode by ID.
- **`best_vnode_for(&self, id: NodeId) -> &VirtualNode`**: Finds the local vnode closest to `id` along the ring.
- **`neighbor_set(&self) -> HashSet<String>`**: Collects all unique peer addresses from predecessor and successor lists across all hosted vnodes.
