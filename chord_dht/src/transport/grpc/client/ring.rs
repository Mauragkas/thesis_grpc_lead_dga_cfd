use async_trait::async_trait;
use tonic::transport::Channel;

use crate::ring::NodeAddr;
use crate::transport::RemoteNode;

use super::super::gen::chord_client::ChordClient;
use super::super::gen::{Empty, FindSuccRequest, GetPredRequest, NotifyRequest, VidMsg};
use super::convert::{from_proto, to_proto};
use super::{GrpcRemote, RPC_TIMEOUT};

#[async_trait]
impl RemoteNode for GrpcRemote {
    async fn find_successor(&self, addr: &str, vid: u64, id: u64) -> Option<NodeAddr> {
        let mut c: ChordClient<Channel> = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.find_successor(FindSuccRequest { vid, id }))
            .await
            .ok()?
            .ok()?;
        Some(from_proto(resp.into_inner()))
    }

    async fn get_predecessor(&self, addr: &str, vid: u64) -> Option<NodeAddr> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.get_predecessor(GetPredRequest { vid }))
            .await
            .ok()?
            .ok()?;
        resp.into_inner().node.map(from_proto)
    }

    async fn get_successor(&self, addr: &str, vid: u64) -> Option<NodeAddr> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(RPC_TIMEOUT, c.get_successor(VidMsg { vid }))
            .await
            .ok()?
            .ok()?;
        Some(from_proto(resp.into_inner()))
    }

    async fn get_successor_list(&self, addr: &str, vid: u64) -> Vec<NodeAddr> {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return Vec::new(),
        };
        match tokio::time::timeout(RPC_TIMEOUT, c.get_successor_list(VidMsg { vid })).await {
            Ok(Ok(r)) => r.into_inner().nodes.into_iter().map(from_proto).collect(),
            _ => Vec::new(),
        }
    }

    async fn notify(&self, addr: &str, vid: u64, self_info: &NodeAddr) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.notify(NotifyRequest {
                vid,
                other: Some(to_proto(self_info)),
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    async fn ping(&self, addr: &str) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(RPC_TIMEOUT, c.ping(Empty {}))
            .await
            .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
            .unwrap_or(false)
    }
}
