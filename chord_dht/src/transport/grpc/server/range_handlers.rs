use tonic::{Request, Response, Status};

use crate::chord::ChordNode;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::range_resp;
use crate::transport::grpc::gen::{
    BoolMsg, DeliverRangeRequest, RangeForwardRequest, RangeRequest, RangeResponse,
};

pub(super) async fn range_query<S, R>(
    node: &std::sync::Arc<ChordNode<S, R>>,
    req: Request<RangeRequest>,
) -> Result<Response<RangeResponse>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let r = req.into_inner();
    let result = node
        .handle_range_query(&r.start_key, r.count, &r.caller_address, r.model_version)
        .await;
    Ok(Response::new(range_resp(result)))
}

pub(super) async fn range_forward<S, R>(
    node: &std::sync::Arc<ChordNode<S, R>>,
    req: Request<RangeForwardRequest>,
) -> Result<Response<RangeResponse>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let r = req.into_inner();
    let payload: Vec<(String, String)> = r.payload.into_iter().map(|e| (e.key, e.value)).collect();
    let result = node
        .handle_range_forward(
            &r.start_key,
            r.count,
            &r.caller_address,
            r.origin_vid,
            r.model_version,
            payload,
        )
        .await;
    Ok(Response::new(range_resp(result)))
}

pub(super) async fn deliver_range(
    req: Request<DeliverRangeRequest>,
) -> Result<Response<BoolMsg>, Status> {
    let r = req.into_inner();
    tracing::debug!(
        entries = r.entries.len(),
        complete = r.complete,
        "received direct range delivery"
    );
    Ok(Response::new(BoolMsg { ok: true }))
}
