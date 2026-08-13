use std::time::Duration;

use crate::ring::{in_range, NodeAddr, NodeId};
use crate::storage::KeyStore;
use crate::transport::{RangeResult, RemoteNode};

use super::{ChordNode, VirtualNode};

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub async fn owns_key_with_model(&self, key: &str, model: &crate::rmi::RmiModel) -> bool {
        let id = model.predict(key);
        for vnode in &self.vnodes {
            let pred = vnode.predecessor.read().await.clone();
            match pred {
                Some(p) => {
                    if in_range(id, p.id, vnode.vid, true) {
                        return true;
                    }
                }
                None => {
                    if vnode.successor().await.id == vnode.vid {
                        return true;
                    }
                }
            }
        }
        false
    }

    async fn get_model_for_version(&self, version: u64) -> crate::rmi::RmiModel {
        let rmi = self.rmi.read().await;
        if rmi.active.version == version {
            rmi.active.clone()
        } else if let Some(ref update) = rmi.update {
            if update.version == version {
                update.clone()
            } else {
                rmi.active.clone() // fallback
            }
        } else {
            rmi.active.clone()
        }
    }

    pub async fn lookup_target(&self, key: &str) -> NodeAddr {
        let id = self.learned_hash(key).await;
        let vnode = self.best_vnode_for(id);
        match tokio::time::timeout(Duration::from_secs(10), self.find_successor(vnode.vid, id))
            .await
        {
            Ok(r) => r,
            Err(_) => {
                tracing::warn!(
                    "lookup_target timed out for key={key}, using immediate successor as fallback"
                );
                vnode.successor().await
            }
        }
    }

    pub async fn owns_key(&self, key: &str) -> bool {
        let id = self.learned_hash(key).await;
        for vnode in &self.vnodes {
            let pred = vnode.predecessor.read().await.clone();
            match pred {
                Some(p) => {
                    if in_range(id, p.id, vnode.vid, true) {
                        return true;
                    }
                }
                None => {
                    if vnode.successor().await.id == vnode.vid {
                        return true; // alone: own everything
                    }
                }
            }
        }
        false
    }

    /// Find the local vnode that owns the given hash id.
    /// Used by handle_range_query to find which vnode's range contains the start key.
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
    /// Used by handle_range_forward to find the next vnode's range after
    /// the previous node's last key. This is the vnode with the smallest
    /// predecessor VID >= from_id — its predecessor is the owner of the
    /// range containing from_id, so this vnode owns the next adjacent range.
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

    pub async fn range_query(&self, start_key: &str, count: u64, caller: &str) -> RangeResult {
        let model = self.rmi.read().await.active.clone();
        let version = model.version;
        if self.owns_key_with_model(start_key, &model).await {
            self.handle_range_query(start_key, count, caller, version)
                .await
        } else {
            let target = self.lookup_target(start_key).await;
            self.remote
                .range_query(&target.address, start_key, count, caller, version)
                .await
                .unwrap_or_default()
        }
    }

    pub async fn handle_range_query(
        &self,
        start_key: &str,
        count: u64,
        caller: &str,
        model_version: u64,
    ) -> RangeResult {
        let need = count as usize;

        // Read RMI model once for consistent hashing throughout this query
        let model = self.get_model_for_version(model_version).await;
        let id = model.predict(start_key);

        // Find the vnode that owns this key and its hash range (pred, vid]
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

        // Overscan (3×) because the BTreeMap contains keys from ALL local vnodes.
        // Filter to only keys whose learned_hash falls in THIS vnode's range,
        // so we don't return keys from non-adjacent hash ranges.
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

        // Forward to this vnode's successor — it owns the next adjacent
        // hash range, so its filtered scan will produce the next batch
        // of keys in Hilbert order.
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

        // Read RMI model once for consistent hashing
        let model = self.get_model_for_version(model_version).await;
        let from_id = model.predict(from_key);

        // Find the local vnode whose range immediately follows from_id.
        // NOT find_owning_vnode(from_id + 1) — that would look inside the
        // previous vnode's range (on a different node). We want the NEXT
        // range, which starts at the previous owner's VID.
        let (vnode, range_start, range_end) = match self.find_next_vnode(from_id).await {
            Some((v, rs, re)) => (v, rs, re),
            None => {
                // No local vnode owns a range after from_id — return what we have
                return RangeResult {
                    entries: payload,
                    complete: true,
                    next_address: String::new(),
                };
            }
        };

        // Overscan + filter, same as handle_range_query
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
