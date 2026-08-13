use crate::ring::{in_range, NodeAddr, NodeId};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::node::ChordNode;

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
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

        let old_id = old.as_ref().map(|o| o.id).unwrap_or(vnode.vid);
        let snap = self.storage.snapshot().await;
        let to_move: Vec<(String, String)> = {
            let rmi = self.rmi.read().await;
            snap.into_iter()
                .filter(|(k, _)| in_range(rmi.active.predict(k), old_id, other.id, true))
                .collect()
        };
        for (k, v) in to_move {
            if self.remote.put_local(&other.address, &k, &v).await {
                self.storage.remove(&k).await;
            }
        }
    }
}
