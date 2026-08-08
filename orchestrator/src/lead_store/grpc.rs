use crate::lead_store::r#trait::LeadStore;
use crate::proto::chord::chord_client::ChordClient;
use crate::proto::chord::{KeyMsg, PutRoutedRequest};
use tonic::{transport::Channel, Code, Status};
use tracing::{debug, error};

/// gRPC-backed lead store. Talks to any LEAD node; `PutRouted`/`GetRouted`
/// route the key to the responsible node inside the chord ring.
pub struct GrpcLeadStore {
    client: ChordClient<Channel>,
}

impl GrpcLeadStore {
    pub fn new(client: ChordClient<Channel>) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl LeadStore for GrpcLeadStore {
    async fn store_gene(&self, key: &str, value: &str) -> Result<(), Status> {
        let mut c = self.client.clone();
        match c
            .put_routed(PutRoutedRequest {
                key: key.to_string(),
                value: value.to_string(),
            })
            .await
        {
            Ok(_) => {
                debug!("Stored gene key '{key}' via PutRouted");
                Ok(())
            }
            Err(e) => {
                error!("PutRouted failed for key '{key}': {e}");
                Err(e)
            }
        }
    }

    async fn get_gene(&self, key: &str) -> Result<Option<String>, Status> {
        let mut c = self.client.clone();
        match c
            .get_routed(KeyMsg {
                key: key.to_string(),
            })
            .await
        {
            Ok(resp) => {
                debug!("Retrieved gene key '{key}' via GetRouted");
                Ok(Some(resp.into_inner().value))
            }
            Err(e) if e.code() == Code::NotFound => {
                debug!("Gene key '{key}' not found (NotFound)");
                Ok(None)
            }
            Err(e) => {
                error!("GetRouted failed for key '{key}': {e}");
                Err(e)
            }
        }
    }
}
