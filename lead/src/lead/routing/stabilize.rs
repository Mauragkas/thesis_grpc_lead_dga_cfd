use tracing::warn;

use crate::ring::{in_range, NodeAddr};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::node::LeadNode;
use super::super::vnode::VirtualNode;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub async fn stabilize_all(&self) {
        for vnode in &self.vnodes {
            if vnode.pruned.load(std::sync::atomic::Ordering::Relaxed) == 1 {
                continue;
            }
            self.stabilize_vnode(vnode).await;
        }
    }

    async fn stabilize_vnode(&self, vnode: &VirtualNode) {
        self.repair_successor_chain(vnode).await;

        let succ = vnode.successor().await;
        if succ.id == vnode.vid {
            self.self_repair(vnode).await;
            return;
        }

        self.merge_predecessor(vnode, &succ).await;
        self.refresh_successor_list(vnode).await;
    }

    /// Walk forward past dead successors until a live one is found, or we
    /// wrap back to ourselves.
    async fn repair_successor_chain(&self, vnode: &VirtualNode) {
        loop {
            let succ = vnode.successor().await;
            if succ.id == vnode.vid {
                break;
            }
            if self.remote.ping(&succ.address).await {
                break;
            }
            warn!("vnode {} successor {} unreachable", vnode.vid, succ.id);
            let mut list = vnode.successor_list.write().await;
            if !list.is_empty() {
                list.remove(0);
            }
            if list.is_empty() {
                list.push(NodeAddr {
                    id: vnode.vid,
                    address: self.self_uri.clone(),
                });
            }
            vnode.fingers.write().await[0] = list.first().cloned();
        }
    }

    /// When alone (successor is self), adopt a live predecessor as successor.
    async fn self_repair(&self, vnode: &VirtualNode) {
        let pred = vnode.predecessor.read().await.clone();
        if let Some(p) = pred {
            if p.id != vnode.vid && self.remote.ping(&p.address).await {
                let mut list = vnode.successor_list.write().await;
                *list = vec![p.clone()];
                vnode.fingers.write().await[0] = Some(p);
            }
        }
    }

    /// Insert a confirmed live predecessor that falls between us and our
    /// successor, ahead of the current successor.
    async fn merge_predecessor(&self, vnode: &VirtualNode, succ: &NodeAddr) {
        if let Some(x) = self.remote.get_predecessor(&succ.address, succ.id).await {
            if x.id != vnode.vid
                && self.remote.ping(&x.address).await
                && in_range(x.id, vnode.vid, succ.id, false)
            {
                let mut list = vnode.successor_list.write().await;
                list.insert(0, x);
                list.truncate(self.config.successor_list_len);
                vnode.fingers.write().await[0] = list.first().cloned();
            }
        }
    }

    /// Refresh the successor list from the current successor, deduping self.
    async fn refresh_successor_list(&self, vnode: &VirtualNode) {
        let cur = vnode.successor().await;
        if cur.id == vnode.vid {
            return;
        }
        let self_node = NodeAddr {
            id: vnode.vid,
            address: self.self_uri.clone(),
        };
        let _ = self.remote.notify(&cur.address, cur.id, &self_node).await;
        let remote_list = self.remote.get_successor_list(&cur.address, cur.id).await;
        let mut new_list = vec![cur.clone()];
        let max_extra = self.config.successor_list_len.saturating_sub(1);
        for n in remote_list.into_iter().take(max_extra) {
            if n.id != vnode.vid && n.id != cur.id && !new_list.iter().any(|m| m.id == n.id) {
                new_list.push(n);
            }
        }
        *vnode.successor_list.write().await = new_list;
    }
}
