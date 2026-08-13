use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::chord::ChordNode;
use crate::rmi::LeafKind;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use crate::transport::grpc::gen::{
    BoolMsg, HeartbeatMsg, LeafDiff, ModelDiff, ModelDiffRequest, ModelParams, PruneRequest,
    ResourceReport,
};

pub(super) async fn push_model<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<ModelParams>,
) -> Result<Response<BoolMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let r = req.into_inner();
    let ok = node.push_model(r.version, &r.data).await;
    Ok(Response::new(BoolMsg { ok }))
}

pub(super) async fn request_model<S, R>(
    node: &Arc<ChordNode<S, R>>,
) -> Result<Response<ModelParams>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let (version, data) = node.request_model().await;
    Ok(Response::new(ModelParams { version, data }))
}

pub(super) async fn get_model_diff<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<ModelDiffRequest>,
) -> Result<Response<ModelDiff>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let _r = req.into_inner();
    let rmi = node.rmi.read().await;
    let current = rmi.update.as_ref().unwrap_or(&rmi.active);
    let diffs: Vec<LeafDiff> = rmi
        .dirty_leaves
        .iter()
        .map(|&idx| {
            let leaf = current.leaves.get(idx);
            let (w, bias, off) = leaf
                .map(|l| match l {
                    LeafKind::Linear(li) => (li.weight, li.bias, li.anchor.offset),
                    LeafKind::RadixSpline(rs) => (0.0, 0.0, rs.anchor.offset),
                })
                .unwrap_or((0.0, 0.0, 0.0));
            LeafDiff {
                index: idx as u32,
                weight: w,
                bias,
                anchor_offset: off,
            }
        })
        .collect();
    Ok(Response::new(ModelDiff {
        new_version: current.version,
        leaves: diffs,
        n: current.n as u64,
    }))
}

pub(super) async fn heartbeat<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<HeartbeatMsg>,
) -> Result<Response<BoolMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let _h = req.into_inner();
    let ready = node.heartbeat().await;
    Ok(Response::new(BoolMsg { ok: ready }))
}

pub(super) async fn report_resources(
    req: Request<ResourceReport>,
) -> Result<Response<BoolMsg>, Status> {
    let r = req.into_inner();
    tracing::debug!(
        mem = r.memory_bytes,
        cpu = r.cpu_percent,
        keys = r.key_count,
        "resource report"
    );
    Ok(Response::new(BoolMsg { ok: true }))
}

pub(super) async fn prune_vnode<S, R>(
    node: &Arc<ChordNode<S, R>>,
    req: Request<PruneRequest>,
) -> Result<Response<BoolMsg>, Status>
where
    S: KeyStore,
    R: RemoteNode,
{
    let r = req.into_inner();
    if let Some(v) = node.find_vnode(r.vid) {
        v.pruned.store(1, std::sync::atomic::Ordering::Relaxed);
        tracing::info!("pruned vnode {} by request: {}", r.vid, r.reason);
        Ok(Response::new(BoolMsg { ok: true }))
    } else {
        Ok(Response::new(BoolMsg { ok: false }))
    }
}
