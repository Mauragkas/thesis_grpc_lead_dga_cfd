use crate::ring::{finger_start, F};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::super::node::LeadNode;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub async fn fix_fingers_all(&self) {
        for vnode in &self.vnodes {
            if vnode.pruned.load(Ordering::Relaxed) == 1 {
                continue;
            }
            let i = self.finger_fix_idx.fetch_add(1, Ordering::Relaxed) % F;
            if i == 0 {
                continue;
            }
            let start = finger_start(vnode.vid, i);
            let s = match tokio::time::timeout(
                Duration::from_secs(5),
                self.find_successor(vnode.vid, start),
            )
            .await
            {
                Ok(r) => r,
                Err(_) => {
                    tracing::warn!("fix_fingers timeout vid={} i={}", vnode.vid, i);
                    vnode.successor().await
                }
            };
            vnode.fingers.write().await[i] = Some(s);
        }
    }

    pub async fn check_predecessor_all(&self) {
        for vnode in &self.vnodes {
            if vnode.pruned.load(Ordering::Relaxed) == 1 {
                continue;
            }
            let pred = vnode.predecessor.read().await.clone();
            if let Some(p) = pred {
                if !self.remote.ping(&p.address).await {
                    *vnode.predecessor.write().await = None;
                }
            }
        }
    }
}
