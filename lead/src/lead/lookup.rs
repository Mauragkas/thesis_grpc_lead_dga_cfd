use std::collections::HashSet;

use crate::ring::{in_range, NodeId};
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::node::LeadNode;
use super::vnode::VirtualNode;

pub(crate) fn in_range_ring(val: NodeId, start: NodeId, end: NodeId, inclusive_end: bool) -> bool {
    in_range(val, start, end, inclusive_end)
}

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub(crate) fn find_vnode(&self, vid: NodeId) -> Option<&VirtualNode> {
        self.vnodes.iter().find(|v| v.vid == vid)
    }

    pub(crate) fn best_vnode_for(&self, id: NodeId) -> &VirtualNode {
        let mut best = &self.vnodes[0];
        for v in &self.vnodes[1..] {
            if v.vid != best.vid && in_range_ring(v.vid, best.vid, id, true) {
                best = v;
            }
        }
        best
    }

    /// Get immediate neighbors: active predecessor and successor list peers.
    pub(crate) async fn neighbor_set(&self) -> HashSet<String> {
        let mut peers = HashSet::new();
        for v in &self.vnodes {
            let pred = v.predecessor.read().await.clone();
            if let Some(p) = pred {
                if p.address != self.self_uri {
                    peers.insert(p.address.clone());
                }
            }
            for s in v.successor_list.read().await.iter() {
                if s.address != self.self_uri {
                    peers.insert(s.address.clone());
                }
            }
        }
        peers
    }
}
