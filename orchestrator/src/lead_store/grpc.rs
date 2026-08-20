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
        let call = c.put_routed(PutRoutedRequest {
            key: key.to_string(),
            value: value.to_string(),
        });
        match tokio::time::timeout(std::time::Duration::from_secs(5), call).await {
            Ok(Ok(_)) => {
                debug!("Stored gene key '{key}' via PutRouted");
                Ok(())
            }
            Ok(Err(e)) => {
                error!("PutRouted failed for key '{key}': {e}");
                Err(e)
            }
            Err(_) => {
                error!("PutRouted timed out after 5s for key '{key}'");
                Err(Status::deadline_exceeded("put_routed timed out"))
            }
        }
    }

    async fn get_gene(&self, key: &str) -> Result<Option<String>, Status> {
        let mut c = self.client.clone();
        let call = c.get_routed(KeyMsg {
            key: key.to_string(),
        });
        match tokio::time::timeout(std::time::Duration::from_secs(5), call).await {
            Ok(Ok(resp)) => {
                debug!("Retrieved gene key '{key}' via GetRouted");
                Ok(Some(resp.into_inner().value))
            }
            Ok(Err(e)) if e.code() == Code::NotFound => {
                debug!("Gene key '{key}' not found (NotFound)");
                Ok(None)
            }
            Ok(Err(e)) => {
                error!("GetRouted failed for key '{key}': {e}");
                Err(e)
            }
            Err(_) => {
                error!("GetRouted timed out after 5s for key '{key}'");
                Err(Status::deadline_exceeded("get_routed timed out"))
            }
        }
    }

    async fn range_query(
        &self,
        start_key: &str,
        count: u64,
    ) -> Result<Vec<(String, String)>, Status> {
        let mut c = self.client.clone();
        let call = c.range_query(RangeRequest {
            start_key: start_key.to_string(),
            count,
            caller_address: String::new(),
        });
        match tokio::time::timeout(std::time::Duration::from_secs(5), call).await {
            Ok(Ok(resp)) => Ok(resp
                .into_inner()
                .entries
                .into_iter()
                .map(|e| (e.key, e.value))
                .collect()),
            Ok(Err(e)) => {
                error!("range_query gRPC failed: {e}");
                Err(e)
            }
            Err(_) => {
                error!("range_query timed out after 5s");
                Err(Status::deadline_exceeded("range_query timed out"))
            }
        }
    }
}
