mod handle;
mod model;

use std::time::Duration;

use crate::ring::{in_range, NodeAddr};
use crate::storage::KeyStore;
use crate::transport::{RangeResult, RemoteNode};

use super::node::ChordNode;

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
}
