use crate::ring::{NodeAddr, NodeId, M};
use crate::rmi::RmiModel;
use tokio::sync::RwLock;

/// Per-virtual-node ring state. Each VID runs an independent Chord protocol.
pub struct VirtualNode {
    pub vid: NodeId,
    pub predecessor: RwLock<Option<NodeAddr>>,
    pub fingers: RwLock<Vec<Option<NodeAddr>>>,
    pub successor_list: RwLock<Vec<NodeAddr>>,
}

impl VirtualNode {
    pub(crate) fn new(vid: NodeId, self_addr: &str) -> Self {
        let self_node = NodeAddr {
            id: vid,
            address: self_addr.to_string(),
        };
        Self {
            vid,
            predecessor: RwLock::new(None),
            fingers: RwLock::new(vec![Some(self_node.clone()); M]),
            successor_list: RwLock::new(vec![self_node]),
        }
    }

    pub async fn successor(&self) -> NodeAddr {
        self.successor_list.read().await.first().cloned().unwrap()
    }
}

pub struct RmiState {
    pub active: RmiModel,
    pub update: Option<RmiModel>,
    pub drift_new: usize,
    pub update_ready: bool,
}
