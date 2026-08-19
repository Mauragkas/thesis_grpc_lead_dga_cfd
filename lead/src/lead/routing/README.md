# Routing & Maintenance Subsystem Module

Implements Chord ring topology operations, bootstrap joining, and periodic self-healing maintenance.

## Files and Code Structure

### 1. `join.rs`
- **`LeadNode::join(&self, known: &str)`**:
  - For each hosted `VirtualNode`:
    1. Sends `find_successor(known, vnode.vid, vnode.vid)` via gRPC.
    2. Receives responsible `NodeAddr` (`succ`).
    3. Fetches successor list via `get_successor_list(succ.address, succ.id)`.
    4. Populates `vnode.successor_list` with `succ` and backup nodes.
    5. Sets `vnode.fingers[0] = Some(succ)`.
    6. Marks `vnode` active.

---

### 2. `stabilize.rs`
- **`LeadNode` Stabilization Methods**:
  - `pub async fn stabilize_all(&self)`: Iterates non-pruned vnodes and calls `stabilize_vnode`.
  - `async fn stabilize_vnode(&self, vnode: &VirtualNode)`:
    1. `repair_successor_chain(vnode)`: Pings successor; if dead, pops entries from `successor_list` until reaching a live node.
    2. `self_repair(vnode)`: If alone, adopts live predecessor as successor.
    3. `merge_predecessor(vnode, succ)`: Queries successor's predecessor $x$. If $x \in (\text{vid}, \text{succ.id})$, inserts $x$ ahead of current successor.
    4. `refresh_successor_list(vnode)`: Sends `notify` to current successor and fetches updated successor list.

---

### 3. `maintain.rs`
- **`LeadNode` Maintenance Methods**:
  - `pub async fn fix_fingers_all(&self)`:
    - Increments `finger_fix_idx % F` ($i \in [1..17]$).
    - Computes target key $\text{start} = \text{finger\_start}(\text{vid}, i) = \text{vid} + 10^{i-1}$.
    - Calls `find_successor(vnode.vid, start)` with 5-second timeout and updates `vnode.fingers[i]`.
  - `pub async fn check_predecessor_all(&self)`:
    - Pings `vnode.predecessor`. If unreachable, resets predecessor to `None`.

---

### 4. `accessors.rs`
- **`all_peers(&self) -> HashSet<String>`**: Collects all unique node addresses across all hosted vnodes' fingers and successor lists.
- **`owning_vnode_for(&self, id: NodeId) -> Option<NodeId>`**: Returns the hosted VID that owns `id` if any.
