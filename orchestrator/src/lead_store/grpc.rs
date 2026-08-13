use crate::lead_store::r#trait::LeadStore;
use crate::proto::lead::lead_client::LeadClient;
use crate::proto::lead::{KeyMsg, PutRoutedRequest, RangeRequest};
use tonic::{transport::Channel, Code, Status};
use tracing::{debug, error};

/// gRPC-backed lead store. Talks to any LEAD node; `PutRouted`/`GetRouted`
/// route the key to the responsible node inside the lead ring.
pub struct GrpcLeadStore {
    client: LeadClient<Channel>,
}

impl GrpcLeadStore {
    pub fn new(client: LeadClient<Channel>) -> Self {
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

    async fn range_query(
        &self,
        start_key: &str,
        count: u64,
    ) -> Result<Vec<(String, String)>, Status> {
        let mut c = self.client.clone();
        let resp = c
            .range_query(RangeRequest {
                start_key: start_key.to_string(),
                count,
                caller_address: String::new(),
            })
            .await?;
        Ok(resp
            .into_inner()
            .entries
            .into_iter()
            .map(|e| (e.key, e.value))
            .collect())
    }
}
