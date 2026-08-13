use async_trait::async_trait;

use crate::transport::RemoteNode;

use super::super::gen::{Empty, KeyMsg, PutRequest};
use super::{GrpcRemote, RPC_TIMEOUT};

#[async_trait]
impl RemoteNode for GrpcRemote {
    async fn get_local(&self, addr: &str, key: &str) -> Option<String> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RPC_TIMEOUT,
            c.get_local(KeyMsg {
                key: key.to_string(),
            }),
        )
        .await
        .ok()?
        .ok()?;
        Some(resp.into_inner().value)
    }

    async fn put_local(&self, addr: &str, key: &str, val: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.put_local(PutRequest {
                key: key.to_string(),
                value: val.to_string(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn delete_local(&self, addr: &str, key: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.delete_local(KeyMsg {
                key: key.to_string(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn get_keys(&self, addr: &str) -> Option<Vec<String>> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.get_keys(Empty {}))
            .await
            .ok()?
            .ok()?;
        Some(resp.into_inner().keys)
    }
}
