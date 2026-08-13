//! Integration tests for the ring module: hashing, ring-state interval
//! logic, and local ring member behaviour.

use orchestrator::ring::{AddressHasher, LocalRingMember, NodeInfo, RingState, Sha256Hasher};
use std::sync::Arc;

// --- AddressHasher tests ---

#[test]
fn hash_is_deterministic() {
    let h = Sha256Hasher;
    assert_eq!(h.hash("orchestrator:50060"), h.hash("orchestrator:50060"));
}

#[test]
fn different_addresses_hash_differently() {
    let h = Sha256Hasher;
    assert_ne!(h.hash("node1:5000"), h.hash("node2:5000"));
}

// --- RingState interval tests ---

#[test]
fn half_open_no_wrap() {
    assert!(RingState::in_half_open(10, 20, 15));
    assert!(RingState::in_half_open(10, 20, 20));
    assert!(!RingState::in_half_open(10, 20, 10));
    assert!(!RingState::in_half_open(10, 20, 21));
}

#[test]
fn half_open_wrap() {
    assert!(RingState::in_half_open(u64::MAX - 5, 5, u64::MAX));
    assert!(RingState::in_half_open(u64::MAX - 5, 5, 0));
    assert!(RingState::in_half_open(u64::MAX - 5, 5, 5));
    assert!(!RingState::in_half_open(u64::MAX - 5, 5, u64::MAX - 6));
}

// --- LocalRingMember tests ---

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
