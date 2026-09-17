use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use crate::config::Config;
use crate::ring::{peer_hash, NodeAddr, NodeId};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::learning::LearnedIndex;
use super::vnode::VirtualNode;

/// A physical LEAD node hosting `k` virtual nodes.
pub struct LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub self_uri: String,
    pub vnodes: Vec<VirtualNode>,
    pub self_info: NodeAddr,
    pub config: Config,
    pub(crate) storage: Arc<S>,
    pub(crate) remote: Arc<R>,
    pub(crate) learning: LearnedIndex,
    /// Round-robin finger index for `fix_fingers_all`; per-instance so
    /// multiple nodes (and tests) don't share a process-global counter.
    pub(crate) finger_fix_idx: AtomicUsize,
}

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub fn new(config: Config, storage: Arc<S>, remote: Arc<R>) -> Self {
        let self_uri = config.self_uri.clone();
        let k = config.virtual_node_count.max(1);
        let mut vids: Vec<NodeId> = (0..k)
            .map(|i| peer_hash(&format!("{i}|{self_uri}")))
            .collect();
        vids.sort();

        let mut vnodes: Vec<VirtualNode> = vids
            .iter()
            .map(|&vid| VirtualNode::new(vid, &self_uri))
            .collect();

        let n = vnodes.len();
        for (i, vnode) in vnodes.iter_mut().enumerate() {
            let next = vids[(i + 1) % n];
            let next_addr = NodeAddr {
                id: next,
                address: self_uri.clone(),
            };
            *vnode.successor_list.get_mut() = vec![next_addr.clone()];
            vnode.fingers.get_mut()[0] = Some(next_addr);
        }

        let self_info = NodeAddr {
            id: vnodes[0].vid,
            address: self_uri.clone(),
        };
        Self {
            self_uri,
            vnodes,
            self_info,
            config,
            storage,
            remote,
            learning: LearnedIndex::new(),
            finger_fix_idx: AtomicUsize::new(1),
        }
    }

    pub fn storage(&self) -> Arc<S> {
        self.storage.clone()
    }
    pub fn remote(&self) -> Arc<R> {
        self.remote.clone()
    }
    pub fn vnode_count(&self) -> usize {
        self.vnodes.len()
    }

    pub async fn vids(&self) -> Vec<NodeId> {
        self.vnodes.iter().map(|v| v.vid).collect()
    }

    pub async fn learned_hash(&self, key: &str) -> NodeId {
        self.learning.predict(key).await
    }

    pub async fn is_alone(&self) -> bool {
        for v in &self.vnodes {
            if v.successor().await.id != v.vid {
                return false;
            }
        }
        true
    }

    /// Initializes learning state with pre-existing keys from persistent storage.
    pub async fn init_from_storage(&self) {
        let n = self.storage.len().await;
        if n > 0 {
            tracing::info!(initial_keys = n, "restored existing keys from storage");
            self.learning.set_keys_total(n);
        }
    }
}
