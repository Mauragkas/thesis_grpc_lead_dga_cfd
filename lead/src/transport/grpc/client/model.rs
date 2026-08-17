use super::super::gen::{HeartbeatMsg, ModelParams, ModelRequest};
use super::{GrpcRemote, RPC_TIMEOUT};

impl GrpcRemote {
    pub(super) async fn push_model_inner(&self, addr: &str, version: u64, data: &[u8]) -> bool {
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

    pub(super) async fn request_model_inner(
        &self,
        addr: &str,
        coordinator: &str,
    ) -> Option<(u64, Vec<u8>)> {
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

    pub(super) async fn heartbeat_inner(
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
