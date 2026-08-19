# gRPC Server Module

Server adapter hosting the `lead.Lead` service via tonic.

## Files and Code Structure

### 1. `mod.rs`
- **`pub async fn serve<S, R>(lead: Arc<LeadNode<S, R>>, addr: &str) -> Result<(), tonic::transport::Error>`**:
  - Parses socket address `addr`.
  - Instantiates `LeadServer::new(lead)`.
  - Starts tonic server listening on `addr`.

---

### 2. `handlers.rs`
- **`LeadServer<S: KeyStore, R: RemoteNode>`** (struct):
  - `lead: Arc<LeadNode<S, R>>`
- **Implemented gRPC Handlers (`lead.Lead` trait)**:
  - `find_successor(request)` $\to$ `lead.find_successor(req.vid, req.id)`
  - `get_predecessor(request)` $\to$ `lead.get_predecessor(req.vid)`
  - `get_successor(request)` $\to$ `lead.get_successor(req.vid)`
  - `get_successor_list(request)` $\to$ `lead.get_successor_list(req.vid)`
  - `notify(request)` $\to$ `lead.notify(req.vid, &from_proto_addr(req.node))`
  - `get_local(request)` $\to$ `lead.storage().get(&req.key)`
  - `put_local(request)` $\to$ `lead.storage().put(req.key, req.val)`
  - `delete_local(request)` $\to$ `lead.storage().remove(&req.key)`
  - `get_keys(request)` $\to$ `lead.storage().snapshot()`
  - `range_query(request)` $\to$ `lead.handle_range_query(&req.start_key, req.count, &req.caller_address, req.model_version)`
  - `range_forward(request)` $\to$ `lead.handle_range_forward(&req.from_key, req.count, &req.caller_address, req.origin_vid, req.model_version, payload)`
  - `push_model(request)` $\to$ `lead.accept_pushed_model(req.version, &req.model_data)`
  - `request_model(request)` $\to$ `lead.active_model()`
  - `heartbeat(request)` $\to$ `lead.is_update_ready()`
  - `put_routed(request)` $\to$ `lead.learned_hash(&req.key)` + `lead.find_successor()` + forward/local put
  - `get_routed(request)` $\to$ `lead.learned_hash(&req.key)` + `lead.find_successor()` + forward/local get
  - `ping(request)` $\to$ returns `{ ok: true }`
