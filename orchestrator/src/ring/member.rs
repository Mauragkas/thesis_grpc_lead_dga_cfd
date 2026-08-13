//! SRP: local ring logic — answers Ring RPCs using `RingState`.
//! This is the "server-side" decision maker; it does no networking.

use crate::ring::state::{NodeInfo, RingState};
use std::sync::Arc;

pub struct LocalRingMember {
    state: Arc<RingState>,
}

impl LocalRingMember {
    pub fn new(state: Arc<RingState>) -> Self {
        Self { state }
    }

    pub fn state(&self) -> &Arc<RingState> {
        &self.state
    }

    /// Find the successor of `id` from this node's perspective.
    /// If we have no successor or the id falls to us, return self.
    /// If the id is in (self.id, successor.id], return successor.
    /// Otherwise, return successor (caller can forward — for small
    /// rings, the successor is the next hop).
    pub async fn find_successor(&self, id: u64) -> NodeInfo {
        let succ = self.state.successor.lock().await.clone();
        let self_id = self.state.self_node.id;

        match succ {
            None => self.state.self_node.clone(),
            Some(s) if s.id == self_id => self.state.self_node.clone(),
            Some(s) => {
                if RingState::in_half_open(self_id, s.id, id) {
                    s
                } else {
                    // Forward to successor. For small rings this is O(n);
                    // a finger table would make it O(log n) — add later
                    // if the orchestrator ring grows large.
                    s
                }
            }
        }
    }

    pub async fn get_predecessor(&self) -> Option<NodeInfo> {
        self.state.predecessor.lock().await.clone()
    }

    /// `notify(other)`: `other` thinks it is our predecessor. Accept
    /// if it's closer to us than our current predecessor.
    pub async fn notify(&self, other: NodeInfo) -> bool {
        let mut pred = self.state.predecessor.lock().await;
        let self_id = self.state.self_node.id;
        let should_accept = match &*pred {
            None => true,
            Some(cur) => RingState::in_open(cur.id, self_id, other.id),
        };
        if should_accept {
            *pred = Some(other);
            true
        } else {
            false
        }
    }

    pub async fn get_successor_list(&self) -> Vec<NodeInfo> {
        self.state.successor_list.lock().await.clone()
    }

    pub async fn set_successor(&self, node: NodeInfo) {
        *self.state.successor.lock().await = Some(node);
    }

    pub async fn set_successor_list(&self, list: Vec<NodeInfo>) {
        *self.state.successor_list.lock().await = list;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(id: u64) -> Arc<RingState> {
        Arc::new(RingState::new(NodeInfo::new(id, format!("node{id}"))))
    }

    #[tokio::test]
    async fn find_successor_returns_self_when_alone() {
        let state = make_state(100);
        let member = LocalRingMember::new(state);
        let s = member.find_successor(50).await;
        assert_eq!(s.id, 100);
    }

    #[tokio::test]
    async fn notify_accepts_closer_predecessor() {
        let state = make_state(100);
        let member = LocalRingMember::new(state.clone());
        // First notify always accepted.
        assert!(member.notify(NodeInfo::new(50, "node50")).await);
        // A node further away (10 is further from 100 than 50) rejected.
        assert!(!member.notify(NodeInfo::new(10, "node10")).await);
        // A node closer (80 is closer to 100 than 50) accepted.
        assert!(member.notify(NodeInfo::new(80, "node80")).await);
        assert_eq!(member.get_predecessor().await.unwrap().id, 80);
    }
}
