use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::ring::{finger_start, hash, in_range, NodeAddr, NodeId, M};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

static FINGER_IDX: AtomicUsize = AtomicUsize::new(1);

/// The chord algorithm. Knows nothing about `HashMap` or transport details —
/// only the `KeyStore` and `RemoteNode` abstractions (DIP).
pub struct ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub self_info: NodeAddr,
    predecessor: RwLock<Option<NodeAddr>>,
    fingers: RwLock<Vec<Option<NodeAddr>>>, // fingers[0] == successor
    storage: Arc<S>,
    remote: Arc<R>,
}

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub fn new(self_info: NodeAddr, storage: Arc<S>, remote: Arc<R>) -> Self {
        Self {
            self_info: self_info.clone(),
            predecessor: RwLock::new(None),
            fingers: RwLock::new(vec![Some(self_info); M]),
            storage,
            remote,
        }
    }

    pub fn storage(&self) -> Arc<S> {
        self.storage.clone()
    }

    pub fn remote(&self) -> Arc<R> {
        self.remote.clone()
    }

    pub async fn successor(&self) -> NodeAddr {
        self.fingers
            .read()
            .await
            .get(0)
            .cloned()
            .flatten()
            .unwrap_or_else(|| self.self_info.clone())
    }

    async fn set_successor(&self, addr: NodeAddr) {
        self.fingers.write().await[0] = Some(addr);
    }

    pub async fn find_successor(&self, id: NodeId) -> NodeAddr {
        let succ = self.successor().await;
        if in_range(id, self.self_info.id, succ.id, true) {
            return succ;
        }
        let n = self.closest_preceding_node(id).await;
        if n.id == self.self_info.id {
            // No better hop available; return our successor.
            return self.successor().await;
        }
        self.remote
            .find_successor(&n.address, id)
            .await
            .unwrap_or(succ)
    }

    async fn closest_preceding_node(&self, id: NodeId) -> NodeAddr {
        let fingers = self.fingers.read().await;
        for i in (0..M).rev() {
            if let Some(f) = &fingers[i] {
                if in_range(f.id, self.self_info.id, id, false) {
                    return f.clone();
                }
            }
        }
        self.self_info.clone()
    }

    pub async fn join(&self, known: &str) {
        info!("join via {}", known);
        if let Some(succ) = self.remote.find_successor(known, self.self_info.id).await {
            info!("joined, successor={}", succ.id);
            self.set_successor(succ).await;
        }
    }

    pub async fn stabilize(&self) {
        let succ = self.successor().await;
        if succ.id == self.self_info.id {
            return;
        }
        if let Some(x) = self.remote.get_predecessor(&succ.address).await {
            if in_range(x.id, self.self_info.id, succ.id, false) {
                self.set_successor(x).await;
            }
        }
        let cur = self.successor().await;
        if cur.id != self.self_info.id {
            let _ = self.remote.notify(&cur.address, &self.self_info).await;
        }
    }

    pub async fn fix_fingers(&self) {
        let i = FINGER_IDX.fetch_add(1, Ordering::Relaxed) % M;
        if i == 0 {
            return; // successor, handled by stabilize
        }
        let start = finger_start(self.self_info.id, i);
        let s = self.find_successor(start).await;
        self.fingers.write().await[i] = Some(s);
    }

    pub async fn check_predecessor(&self) {
        let pred = self.predecessor.read().await.clone();
        if let Some(p) = pred {
            if !self.remote.ping(&p.address).await {
                warn!("predecessor {} unreachable", p.id);
                *self.predecessor.write().await = None;
            }
        }
    }

    pub async fn predecessor(&self) -> Option<NodeAddr> {
        self.predecessor.read().await.clone()
    }

    /// Called when `other` claims to be our predecessor. Updates predecessor
    /// and migrates any keys that now belong to `other`.
    pub async fn notify(&self, other: NodeAddr) {
        let mut pred = self.predecessor.write().await;
        let old = pred.clone();
        let should_update =
            old.is_none() || in_range(other.id, old.as_ref().unwrap().id, self.self_info.id, false);
        if !should_update {
            return;
        }
        *pred = Some(other.clone());
        drop(pred);

        // Keys in (old_or_self, other] now belong to `other`.
        let old_id = old.as_ref().map(|o| o.id).unwrap_or(self.self_info.id);
        let to_move = self.storage.keys_in_range(old_id, other.id, true).await;

        for (k, v) in to_move {
            if self.remote.put_local(&other.address, &k, &v).await {
                self.storage.remove(&k).await;
            }
        }
    }

    /// Convenience: locate the node responsible for a key.
    pub async fn lookup_target(&self, key: &str) -> NodeAddr {
        self.find_successor(hash(key)).await
    }
}
