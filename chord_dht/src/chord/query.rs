use std::time::Duration;

use crate::ring::{in_range, NodeAddr};
use crate::storage::KeyStore;
use crate::transport::{RangeResult, RemoteNode};

use super::ChordNode;

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
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

    pub async fn range_query(&self, start_key: &str, count: u64, caller: &str) -> RangeResult {
        if self.owns_key(start_key).await {
            self.handle_range_query(start_key, count, caller).await
        } else {
            let target = self.lookup_target(start_key).await;
            self.remote
                .range_query(&target.address, start_key, count, caller)
                .await
                .unwrap_or_default()
        }
    }

    pub async fn handle_range_query(
        &self,
        start_key: &str,
        count: u64,
        caller: &str,
    ) -> RangeResult {
        let need = count as usize;
        let id = self.learned_hash(start_key).await;
        let vnode = self.best_vnode_for(id);
        let origin_vid = vnode.vid;

        let local = self.storage.range_scan(start_key, need).await;
        let payload = local;

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
        mut payload: Vec<(String, String)>,
    ) -> RangeResult {
        let need = count as usize;
        let local = self.storage.range_scan_after(from_key, need).await;
        payload.extend(local);

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
        let id = self.learned_hash(&last_key).await;
        let vnode = self.best_vnode_for(id);
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
