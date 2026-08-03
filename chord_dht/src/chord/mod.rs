use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::ring::{peer_hash, NodeAddr, NodeId};
use crate::rmi::RmiModel;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

pub mod learning;
pub mod query;
pub mod routing;
pub mod vnode;

pub use vnode::{RmiState, VirtualNode};

pub const R: usize = 4; // successor-list length
pub const DEFAULT_K: usize = 10; // virtual nodes per physical node

/// A physical LEAD node hosting `k` virtual nodes.
pub struct ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub self_uri: String,
    pub vnodes: Vec<VirtualNode>,
    pub self_info: NodeAddr,
    pub(crate) storage: Arc<S>,
    pub(crate) remote: Arc<R>,
    pub(crate) rmi: RwLock<RmiState>,
    pub(crate) keys_total: AtomicUsize,
    pub(crate) model_version_counter: AtomicU64,
}

impl<S, R> ChordNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub fn new(self_uri: String, k: usize, storage: Arc<S>, remote: Arc<R>) -> Self {
        let k = k.max(1);

        let mut vids: Vec<NodeId> = (0..k)
            .map(|i| peer_hash(&format!("{i}|{self_uri}")))
            .collect();
        vids.sort();

        let mut vnodes: Vec<VirtualNode> = vids
            .iter()
            .map(|&vid| VirtualNode::new(vid, &self_uri))
            .collect();

        // Link siblings into an initial ring so the node starts in a valid state
        // instead of 10 disconnected "alone" vnodes.
        let n = vnodes.len();
        for (i, vnode) in vnodes.iter_mut().enumerate() {
            let next = vids[(i + 1) % n];
            let prev = vids[(i + n - 1) % n];
            let next_addr = NodeAddr {
                id: next,
                address: self_uri.clone(),
            };
            *vnode.successor_list.get_mut() = vec![next_addr.clone()];
            vnode.fingers.get_mut()[0] = Some(next_addr);
            *vnode.predecessor.get_mut() = Some(NodeAddr {
                id: prev,
                address: self_uri.clone(),
            });
        }

        let self_info = NodeAddr {
            id: vnodes[0].vid,
            address: self_uri.clone(),
        };
        Self {
            self_uri: self_uri.clone(),
            vnodes,
            self_info,
            storage,
            remote,
            rmi: RwLock::new(RmiState {
                active: RmiModel::default(),
                update: None,
                drift_new: 0,
                update_ready: false,
            }),
            keys_total: AtomicUsize::new(0),
            model_version_counter: AtomicU64::new(1),
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
        self.rmi.read().await.active.predict(key)
    }

    pub async fn is_alone(&self) -> bool {
        for v in &self.vnodes {
            if v.successor().await.id != v.vid {
                return false;
            }
        }
        true
    }

    pub(crate) fn find_vnode(&self, vid: NodeId) -> Option<&VirtualNode> {
        self.vnodes.iter().find(|v| v.vid == vid)
    }

    /// Pick the vnode whose VID is closest preceding `id` (for initiating lookups).
    pub(crate) fn best_vnode_for(&self, id: NodeId) -> &VirtualNode {
        let mut best = &self.vnodes[0];
        for v in &self.vnodes[1..] {
            if v.vid != best.vid && in_range_ring(v.vid, best.vid, id, true) {
                best = v;
            }
        }
        best
    }
}

fn in_range_ring(val: NodeId, start: NodeId, end: NodeId, inclusive_end: bool) -> bool {
    crate::ring::in_range(val, start, end, inclusive_end)
}
