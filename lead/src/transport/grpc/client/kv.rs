use super::super::gen::{Empty, KeyMsg, PutRequest};
use super::{GrpcRemote, RPC_TIMEOUT};

impl GrpcRemote {
    pub(super) async fn get_local_inner(&self, addr: &str, key: &str) -> Option<String> {
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

    pub(super) async fn put_local_inner(&self, addr: &str, key: &str, val: &str) -> bool {
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

    pub(super) async fn delete_local_inner(&self, addr: &str, key: &str) -> bool {
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

    pub(super) async fn get_keys_inner(&self, addr: &str) -> Option<Vec<String>> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.get_keys(Empty {}))
            .await
            .ok()?
            .ok()?;
        Some(resp.into_inner().keys)
    }
}
