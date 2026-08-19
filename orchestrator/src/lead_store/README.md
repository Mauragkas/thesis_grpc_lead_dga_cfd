# LEAD Store Module

Client adapter for persisting and retrieving gene payloads in the distributed LEAD DHT.

## Files and Code Structure

### 1. `trait.rs`
- **`LeadStore`** (async trait):
  ```rust
  #[async_trait::async_trait]
  pub trait LeadStore: Send + Sync {
      async fn store_gene(&self, key: &str, value: &str) -> Result<(), tonic::Status>;
      async fn get_gene(&self, key: &str) -> Result<Option<String>, tonic::Status>;
      async fn range_query(&self, start_key: &str, count: u64) -> Result<Vec<(String, String)>, tonic::Status>;
  }
  ```

---

### 2. `mod.rs`
- **`GenePayload`** (struct, serde Serialize/Deserialize):
  - `pub genes: Vec<f64>`: Gene vector.
  - `pub fitness: f64`: Evaluated fitness score.
  - `pub generation: usize`: Origin generation.
- **`pub fn gene_key(genes: &[f64]) -> String`**:
  - Deterministic JSON representation used as exact key.

---

### 3. `grpc.rs`
- **`GrpcLeadStore`**:
  - **Fields**:
    - `client: LeadClient<Channel>`: Tonic client for the `lead.Lead` gRPC service.
  - **Methods**:
    - `pub fn new(client: LeadClient<Channel>) -> Self`: Constructor.
    - `store_gene(&self, key: &str, value: &str) -> Result<(), Status>`: Calls `/lead.Lead/PutRouted` to route and store key-value pair.
    - `get_gene(&self, key: &str) -> Result<Option<String>, Status>`: Calls `/lead.Lead/GetRouted` to retrieve stored payload.
    - `range_query(&self, start_key: &str, count: u64) -> Result<Vec<(String, String)>, Status>`: Calls `/lead.Lead/RangeQuery` to perform order-preserving DHT range scans.
