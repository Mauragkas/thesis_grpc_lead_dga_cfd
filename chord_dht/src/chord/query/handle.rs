use crate::ring::{in_range, NodeId};
use crate::storage::KeyStore;
use crate::transport::{RangeResult, RemoteNode};

use super::super::node::ChordNode;
use super::super::vnode::VirtualNode;

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    /// Find the local vnode that owns the given hash id.
    async fn find_owning_vnode(&self, id: NodeId) -> Option<(&VirtualNode, NodeId, NodeId)> {
        for v in &self.vnodes {
            let pred = v.predecessor.read().await;
            match pred.as_ref() {
                Some(p) => {
                    if in_range(id, p.id, v.vid, true) {
                        return Some((v, p.id, v.vid));
                    }
                }
                None => {
                    if v.successor().await.id == v.vid {
                        return Some((v, 0, v.vid));
                    }
                }
            }
        }
        None
    }

    /// Find the local vnode whose range immediately FOLLOWS from_id.
    async fn find_next_vnode(&self, from_id: NodeId) -> Option<(&VirtualNode, NodeId, NodeId)> {
        let mut best: Option<(&VirtualNode, NodeId, NodeId)> = None;
        for v in &self.vnodes {
            let pred = v.predecessor.read().await;
            if let Some(p) = pred.as_ref() {
                if p.id >= from_id {
                    if best.is_none() || p.id < best.as_ref().unwrap().1 {
                        best = Some((v, p.id, v.vid));
                    }
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

        let (vnode, range_start, range_end) = match self.find_owning_vnode(id).await {
            Some((v, rs, re)) => (v, rs, re),
            None => {
                let v = self.best_vnode_for(id);
                let pred = v.predecessor.read().await;
                let rs = pred.as_ref().map(|p| p.id).unwrap_or(0);
                (v, rs, v.vid)
            }
        };
        let origin_vid = vnode.vid;

        let overscan = need.saturating_mul(3).max(need);
        let local = self.storage.range_scan(start_key, overscan).await;
        let payload: Vec<(String, String)> = local
            .into_iter()
            .filter(|(k, _)| {
                let h = model.predict(k);
                in_range(h, range_start, range_end, true)
            })
            .take(need)
            .collect();

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

        let (vnode, range_start, range_end) = match self.find_next_vnode(from_id).await {
            Some((v, rs, re)) => (v, rs, re),
            None => {
                return RangeResult {
                    entries: payload,
                    complete: true,
                    next_address: String::new(),
                };
            }
        };

        let overscan = need.saturating_mul(3).max(need);
        let local = self.storage.range_scan_after(from_key, overscan).await;
        let filtered: Vec<(String, String)> = local
            .into_iter()
            .filter(|(k, _)| {
                let h = model.predict(k);
                in_range(h, range_start, range_end, true)
            })
            .take(need)
            .collect();
        payload.extend(filtered);

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
            .unwrap_or_else(|| from_key.to_string());

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
