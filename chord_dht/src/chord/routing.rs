use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tracing::{info, warn};

use crate::ring::{finger_start, in_range, NodeAddr, NodeId, M};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::{ChordNode, VirtualNode, R};

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
        let succ = vnode.successor().await;
        if in_range(id, vnode.vid, succ.id, true) {
            return succ;
        }
        let n = self.closest_preceding(vnode, id).await;
        if n.id == vnode.vid {
            return vnode.successor().await;
        }
        match self.remote.find_successor(&n.address, n.id, id).await {
            Some(r) => r,
            None => {
                // Safety rule 3: Chord fallback forwarding on RMI/remote failure.
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
        for i in (0..M).rev() {
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

    pub async fn join(&self, known: &str) {
        for vnode in &self.vnodes {
            if let Some(succ) = self
                .remote
                .find_successor(known, vnode.vid, vnode.vid)
                .await
            {
                let mut list = vec![succ.clone()];
                let remote_list = self.remote.get_successor_list(&succ.address, succ.id).await;
                for n in remote_list.into_iter().take(R - 1) {
                    if n.id != succ.id && !list.contains(&n) {
                        list.push(n);
                    }
                }
                *vnode.successor_list.write().await = list;
                vnode.fingers.write().await[0] = Some(succ);
            }
        }
        info!("joined ring with {} vnodes", self.vnodes.len());
    }

    pub async fn stabilize_all(&self) {
        for vnode in &self.vnodes {
            self.stabilize_vnode(vnode).await;
        }
    }

    async fn stabilize_vnode(&self, vnode: &VirtualNode) {
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

        let succ = vnode.successor().await;
        if succ.id == vnode.vid {
            let pred = vnode.predecessor.read().await.clone();
            if let Some(p) = pred {
                if p.id != vnode.vid && self.remote.ping(&p.address).await {
                    let mut list = vnode.successor_list.write().await;
                    *list = vec![p.clone()];
                    vnode.fingers.write().await[0] = Some(p);
                }
            }
            return;
        }

        if let Some(x) = self.remote.get_predecessor(&succ.address, succ.id).await {
            if x.id != vnode.vid
                && self.remote.ping(&x.address).await
                && in_range(x.id, vnode.vid, succ.id, false)
            {
                let mut list = vnode.successor_list.write().await;
                list.insert(0, x);
                list.truncate(R);
                vnode.fingers.write().await[0] = list.first().cloned();
            }
        }

        let cur = vnode.successor().await;
        if cur.id != vnode.vid {
            let self_node = NodeAddr {
                id: vnode.vid,
                address: self.self_uri.clone(),
            };
            let _ = self.remote.notify(&cur.address, cur.id, &self_node).await;
            let remote_list = self.remote.get_successor_list(&cur.address, cur.id).await;
            let mut new_list = vec![cur.clone()];
            for n in remote_list.into_iter().take(R - 1) {
                if n.id != vnode.vid && n.id != cur.id && !new_list.iter().any(|m| m.id == n.id) {
                    new_list.push(n);
                }
            }
            *vnode.successor_list.write().await = new_list;
        }
    }

    pub async fn fix_fingers_all(&self) {
        static IDX: AtomicUsize = AtomicUsize::new(1);
        for vnode in &self.vnodes {
            let i = IDX.fetch_add(1, Ordering::Relaxed) % M;
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
            let pred = vnode.predecessor.read().await.clone();
            if let Some(p) = pred {
                if !self.remote.ping(&p.address).await {
                    *vnode.predecessor.write().await = None;
                }
            }
        }
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
