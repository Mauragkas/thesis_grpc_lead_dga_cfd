# gRPC Client Module

Client adapter for communicating with remote LEAD DHT nodes.

## Files and Code Structure

### 1. `mod.rs`
- **`GrpcRemote`** (struct):
  - **Fields**:
    - `channels: Mutex<HashMap<String, Channel>>`: Connection pool cache keyed by target address.
  - **Methods**:
    - `pub fn new() -> Self`: Constructor.
    - `async fn channel(&self, addr: &str) -> Result<Channel, tonic::Status>`: Returns cached channel or initiates new connection.

---

### 2. `ring.rs`
- Implements `RingClient` for `GrpcRemote`:
  - `find_successor`: Calls `/lead.Lead/FindSuccessor`.
  - `get_predecessor`: Calls `/lead.Lead/GetPredecessor`.
  - `get_successor`: Calls `/lead.Lead/GetSuccessor`.
  - `get_successor_list`: Calls `/lead.Lead/GetSuccessorList`.
  - `notify`: Calls `/lead.Lead/Notify`.
  - `ping`: Calls `/lead.Lead/Ping`.

---

### 3. `kv.rs`
- Implements `KvClient` for `GrpcRemote`:
  - `get_local`: Calls `/lead.Lead/GetLocal`.
  - `put_local`: Calls `/lead.Lead/PutLocal`.
  - `delete_local`: Calls `/lead.Lead/DeleteLocal`.
  - `get_keys`: Calls `/lead.Lead/GetKeys`.

---

### 4. `range.rs`
- Implements `RangeClient` for `GrpcRemote`:
  - `range_query`: Calls `/lead.Lead/RangeQuery`.
  - `range_forward`: Calls `/lead.Lead/RangeForward`.

---

### 5. `model.rs`
- Implements `ModelClient` for `GrpcRemote`:
  - `push_model`: Calls `/lead.Lead/PushModel`.
  - `request_model`: Calls `/lead.Lead/RequestModel`.
  - `heartbeat`: Calls `/lead.Lead/Heartbeat`.

---

### 6. `convert.rs`
- Data conversion helpers:
  - `pub fn to_proto_addr(addr: &NodeAddr) -> lead_proto::NodeAddr`
  - `pub fn from_proto_addr(addr: lead_proto::NodeAddr) -> NodeAddr`
  - `pub fn from_proto_entries(entries: Vec<lead_proto::KvPair>) -> Vec<(String, String)>`
