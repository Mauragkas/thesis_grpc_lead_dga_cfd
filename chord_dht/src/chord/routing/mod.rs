mod accessors;
mod join;
mod maintain;
mod stabilize;

use tracing::warn;

use crate::ring::{in_range, NodeAddr, NodeId, F};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::node::ChordNode;
use super::vnode::VirtualNode;
use super::R;

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub async fn find_successor(&self, vid: NodeId, id: NodeId) -> NodeAddr {
        let vnode = if let Some(v) = self.find_vnode(vid) {
            v
        } else {
            self.best_vnode_for(id)
        };
        vnode.record_request();
        if vnode.pruned.load(std::sync::atomic::Ordering::Relaxed) == 1 {
            vnode.record_error();
            return vnode.successor().await;
        }
        let succ = vnode.successor().await;
        if in_range(id, vnode.vid, succ.id, true) {
            return succ;
        }
        let n = self.closest_preceding(vnode, id).await;
        if n.id == vnode.vid {
            return vnode.successor().await;
        }
        match self.remote.find_successor(&n.address, n.id, id).await {
            Some(r) => {
                vnode.mark_active().await;
                r
            }
            None => {
                vnode.record_error();
                warn!("find_successor remote failed vid={vid} id={id}, fallback");
                self.fallback_find_successor(vnode, id).await
            }
        }
    }

    async fn fallback_find_successor(&self, vnode: &VirtualNode, id: NodeId) -> NodeAddr {
        let fingers = vnode.fingers.read().await.clone();
        for f in fingers.into_iter().flatten() {
            if in_range(f.id, vnode.vid, id, false) {
                if let Some(r) = self.remote.find_successor(&f.address, f.id, id).await {
                    return r;
                }
            }
        }
        vnode.successor().await
    }

    async fn closest_preceding(&self, vnode: &VirtualNode, id: NodeId) -> NodeAddr {
        let fingers = vnode.fingers.read().await;
        for i in (0..F).rev() {
            if let Some(f) = &fingers[i] {
                if in_range(f.id, vnode.vid, id, false) {
                    return f.clone();
                }
            }
        }
        NodeAddr {
            id: vnode.vid,
            address: self.self_uri.clone(),
        }
    }
}
