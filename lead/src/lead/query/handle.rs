use crate::ring::NodeId;
use crate::storage::KeyStore;
use crate::transport::{RangeResult, RemoteNode};

use super::super::node::LeadNode;
use super::super::vnode::VirtualNode;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    /// Find the local vnode that owns the given hash id.
    async fn find_owning_vnode(&self, id: NodeId) -> Option<(&VirtualNode, NodeId, NodeId)> {
        let vid = self.owning_vnode_for(id).await?;
        let v = self.find_vnode(vid)?;
        let pred = v.predecessor.read().await;
        let rs = pred.as_ref().map(|p| p.id).unwrap_or(0);
        Some((v, rs, vid))
    }

    /// Find the local vnode whose range immediately FOLLOWS from_id.
    async fn find_next_vnode(&self, from_id: NodeId) -> Option<(&VirtualNode, NodeId, NodeId)> {
        let mut best: Option<(&VirtualNode, NodeId, NodeId)> = None;
        for v in &self.vnodes {
            let pred = v.predecessor.read().await;
            if let Some(p) = pred.as_ref() {
                if p.id >= from_id && (best.is_none() || p.id < best.as_ref().unwrap().1) {
                    best = Some((v, p.id, v.vid));
                }
            }
        }
        best
    }

    pub async fn handle_range_query(
        &self,
        start_key: &str,
        count: u64,
        caller: &str,
        model_version: u64,
    ) -> RangeResult {
        let need = count as usize;
        let model = self.get_model_for_version(model_version).await;
        let id = model.predict(start_key);

        let vnode = match self.find_owning_vnode(id).await {
            Some((v, _, _)) => v,
            None => self.best_vnode_for(id),
        };

        let overscan = need
            .saturating_mul(self.config.range_overscan_multiplier)
            .max(need);
        let local = self.storage.range_scan(start_key, overscan).await;
        let payload = self.collect_owned(local, &model, need).await;
        self.complete_or_forward(payload, need, start_key, vnode, vnode.vid, caller, model_version)
            .await
    }

    pub async fn handle_range_forward(
        &self,
        from_key: &str,
        count: u64,
        caller: &str,
        origin_vid: u64,
        model_version: u64,
        mut payload: Vec<(String, String)>,
    ) -> RangeResult {
        let need = count as usize;
        let model = self.get_model_for_version(model_version).await;
        let from_id = model.predict(from_key);

        let vnode = match self.find_next_vnode(from_id).await {
            Some((v, _, _)) => v,
            None => {
                return RangeResult {
                    entries: payload,
                    complete: true,
                    next_address: String::new(),
                };
            }
        };

        let overscan = need
            .saturating_mul(self.config.range_overscan_multiplier)
            .max(need);
        let local = self.storage.range_scan_after(from_key, overscan).await;
        let filtered = self.collect_owned(local, &model, need).await;
        payload.extend(filtered);
        self.complete_or_forward(payload, need, from_key, vnode, origin_vid, caller, model_version)
            .await
    }

    /// Keep keys from a local scan that are owned by *any* of this node's
    /// vnodes (per the given model), up to `need`. Keys owned by remote nodes
    /// (stale / pending migration) are excluded so the result only contains
    /// keys this node legitimately serves.
    async fn collect_owned(
        &self,
        local: Vec<(String, String)>,
        model: &crate::rmi::RmiModel,
        need: usize,
    ) -> Vec<(String, String)> {
        let mut owned = Vec::with_capacity(need.min(local.len()));
        for (k, v) in local {
            if self.owns_key_with_model(&k, model).await {
                owned.push((k, v));
                if owned.len() >= need {
                    break;
                }
            }
        }
        owned
    }

    /// Shared tail for both range paths: if we have enough entries, return
    /// complete; otherwise forward the remaining request to our successor,
    /// stopping when the ring wraps back to `origin_vid`.
    async fn complete_or_forward(
        &self,
        payload: Vec<(String, String)>,
        need: usize,
        start_key: &str,
        vnode: &VirtualNode,
        origin_vid: u64,
        caller: &str,
        model_version: u64,
    ) -> RangeResult {
        if payload.len() >= need {
            return RangeResult {
                entries: payload,
                complete: true,
                next_address: String::new(),
            };
        }

        let remaining = need - payload.len();
        let last_key = payload
            .last()
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| start_key.to_string());

        let succ = vnode.successor().await;
        if succ.id == vnode.vid || succ.id == origin_vid {
            return RangeResult {
                entries: payload,
                complete: true,
                next_address: String::new(),
            };
        }

        self.remote
            .range_forward(
                &succ.address,
                &last_key,
                remaining as u64,
                caller,
                origin_vid,
                model_version,
                payload.clone(),
            )
            .await
            .unwrap_or(RangeResult {
                entries: payload,
                complete: true,
                next_address: String::new(),
            })
    }
}
