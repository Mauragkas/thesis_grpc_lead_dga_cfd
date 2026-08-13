use crate::transport::RangeResult;

use super::super::gen::{DeliverRangeRequest, RangeForwardRequest, RangeRequest};
use super::convert::{entries_from_proto, entries_to_proto};
use super::{GrpcRemote, RANGE_TIMEOUT, RPC_TIMEOUT};

impl GrpcRemote {
    pub(super) async fn deliver_range_inner(
        &self,
        addr: &str,
        entries: &[(String, String)],
        complete: bool,
    ) -> bool {
        let mut c = match self.client(addr).await {
            Some(c) => c,
            None => return false,
        };
        tokio::time::timeout(
            RPC_TIMEOUT,
            c.deliver_range(DeliverRangeRequest {
                caller_address: addr.to_string(),
                entries: entries_to_proto(entries),
                complete,
            }),
        )
        .await
        .map(|r| r.map(|r| r.into_inner().ok).unwrap_or(false))
        .unwrap_or(false)
    }

    pub(super) async fn range_query_inner(
        &self,
        addr: &str,
        start_key: &str,
        count: u64,
        caller: &str,
        model_version: u64,
    ) -> Option<RangeResult> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RANGE_TIMEOUT,
            c.range_query(RangeRequest {
                start_key: start_key.to_string(),
                count,
                caller_address: caller.to_string(),
                model_version,
            }),
        )
        .await
        .ok()?
        .ok()?;
        let inner = resp.into_inner();
        Some(RangeResult {
            entries: entries_from_proto(inner.entries),
            complete: inner.complete,
            next_address: inner.next_address,
        })
    }

    pub(super) async fn range_forward_inner(
        &self,
        addr: &str,
        from_key: &str,
        count: u64,
        caller: &str,
        origin_vid: u64,
        model_version: u64,
        payload: Vec<(String, String)>,
    ) -> Option<RangeResult> {
        let mut c = self.client(addr).await?;
        let resp = tokio::time::timeout(
            RANGE_TIMEOUT,
            c.range_forward(RangeForwardRequest {
                start_key: from_key.to_string(),
                count,
                caller_address: caller.to_string(),
                payload: entries_to_proto(&payload),
                origin_vid,
                model_version,
            }),
        )
        .await
        .ok()?
        .ok()?;
        let inner = resp.into_inner();
        Some(RangeResult {
            entries: entries_from_proto(inner.entries),
            complete: inner.complete,
            next_address: inner.next_address,
        })
    }
}
