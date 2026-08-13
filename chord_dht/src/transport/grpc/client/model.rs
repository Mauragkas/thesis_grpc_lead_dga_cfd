use async_trait::async_trait;

use crate::transport::RemoteNode;

use super::super::gen::{HeartbeatMsg, ModelParams, ModelRequest, PruneRequest};
use super::{GrpcRemote, RPC_TIMEOUT};

#[async_trait]
impl RemoteNode for GrpcRemote {
    async fn prune_vnode(&self, addr: &str, vid: u64, target_vid: u64, reason: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.prune_vnode(PruneRequest {
                vid,
                target_vid,
                reason: reason.to_string(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn push_model(&self, addr: &str, version: u64, data: &[u8]) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.push_model(ModelParams {
                version,
                data: data.to_vec(),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn request_model(&self, addr: &str, coordinator: &str) -> Option<(u64, Vec<u8>)> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RPC_TIMEOUT,
            c.request_model(ModelRequest {
                coordinator: coordinator.to_string(),
            }),
        )
        .await
        .ok()?
        .ok()?;
        let inner = resp.into_inner();
        Some((inner.version, inner.data))
    }

    async fn heartbeat(
        &self,
        addr: &str,
        sender: &str,
        update_ready: bool,
        model_version: u64,
    ) -> Option<bool> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RPC_TIMEOUT,
            c.heartbeat(HeartbeatMsg {
                sender: sender.to_string(),
                update_ready,
                model_version,
            }),
        )
        .await
        .ok()?
        .ok()?;
        Some(resp.into_inner().ok)
    }
}
