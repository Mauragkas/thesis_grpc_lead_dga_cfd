use crate::ring::{in_range, NodeAddr, NodeId};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::node::LeadNode;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    /// VID of the local vnode whose (predecessor, vid] window contains `id`,
    /// if any. A vnode with no predecessor (alone in the ring) owns everything.
    pub async fn owning_vnode_for(&self, id: NodeId) -> Option<NodeId> {
        for v in &self.vnodes {
            let pred = v.predecessor.read().await.clone();
            match pred {
                Some(p) => {
                    if in_range(id, p.id, v.vid, true) {
                        return Some(v.vid);
                    }
                }
                None => {
                    if v.successor().await.id == v.vid {
                        return Some(v.vid);
                    }
                }
            }
        }
        None
    }

    pub async fn predecessor(&self, vid: NodeId) -> Option<NodeAddr> {
        self.find_vnode(vid)?.predecessor.read().await.clone()
    }

    pub async fn successor(&self, vid: NodeId) -> NodeAddr {
        if let Some(v) = self.find_vnode(vid) {
            v.successor().await
        } else {
            NodeAddr {
                id: 0,
                address: self.self_uri.clone(),
            }
        }
    }

    pub async fn successor_list(&self, vid: NodeId) -> Vec<NodeAddr> {
        if let Some(v) = self.find_vnode(vid) {
            v.successor_list.read().await.clone()
        } else {
            vec![]
        }
    }

    pub async fn notify(&self, vid: NodeId, other: NodeAddr) {
        let vnode = match self.find_vnode(vid) {
            Some(v) => v,
            None => return,
        };
        let mut pred = vnode.predecessor.write().await;
        let old = pred.clone();
        let should =
            old.is_none() || in_range(other.id, old.as_ref().unwrap().id, vnode.vid, false);
        if !should {
            return;
        }
        *pred = Some(other.clone());
        drop(pred);

        // If old was None, we didn't split an existing interval we were previously holding.
        let old = match old {
            Some(o) => o,
            None => return,
        };

        // Don't migrate keys to ourselves
        if other.address == self.self_uri {
            return;
        }

        let snap = self.storage.snapshot().await;
        let mut to_move: Vec<(String, String)> = Vec::new();
        for (k, v) in snap {
            let h = self.learning.predict(&k).await;
            if in_range(h, old.id, other.id, true) && !self.owns_key(&k).await {
                to_move.push((k, v));
            }
        }
        for (k, v) in to_move {
            if self.remote.put_local(&other.address, &k, &v).await {
                self.storage.remove(&k).await;
            }
        }
    }
}
