use lead_node::lead::learning::LearnedIndex;
use lead_node::lead::PID_ADJUST_INTERVAL;
use lead_node::rmi::{feature, RmiModel};

#[test]
fn in_grace_just_after_construction() {
    let li = LearnedIndex::new();
    assert!(li.in_grace(10_000), "large grace should hold at startup");
    assert!(!li.in_grace(0), "zero grace: elapsed >= 0 already past it");
}

#[tokio::test]
async fn record_insert_raises_update_ready_after_threshold() {
    let li = LearnedIndex::new();
    // grace=0, min_keys=50, threshold=0.40; drift_new == keys_total so ratio=1.0.
    for _ in 0..50 {
        let _ = li
            .record_insert("key", 0, 50, 0.40, PID_ADJUST_INTERVAL)
            .await;
    }
    assert!(li.is_update_ready().await);
}

#[tokio::test]
async fn record_insert_marks_dirty_leaf() {
    let li = LearnedIndex::new();
    // grace large so update_ready is not raised; we only care about dirty.
    let _ = li
        .record_insert("somekey", 10_000, 50, 0.40, PID_ADJUST_INTERVAL)
        .await;
    let dirty = li.dirty_leaf_indices().await;
    let bins = li.active_model().await.stage0_bins;
    let f = feature("somekey");
    let bin = ((f * bins as f64) as usize).min(bins.saturating_sub(1));
    assert!(
        dirty.contains(&bin),
        "expected bin {bin} dirty, got {dirty:?}"
    );
}

#[tokio::test]
async fn accept_pushed_model_rejects_rollback_bad_payload_then_accepts() {
    let li = LearnedIndex::new();
    assert_eq!(li.version().await, 1);

    let v2 = serde_json::to_vec(&RmiModel {
        version: 2,
        ..RmiModel::default()
    })
    .unwrap();

    // Rollback: version 1 <= active 1.
    assert!(!li.accept_pushed_model(1, &v2).await);
    assert_eq!(li.version().await, 1);

    // Malformed payload.
    assert!(!li.accept_pushed_model(2, b"not-json").await);
    assert_eq!(li.version().await, 1);

    // Valid push.
    assert!(li.accept_pushed_model(2, &v2).await);
    assert_eq!(li.version().await, 2);
    assert_eq!(li.next_version(), 3);
}

#[tokio::test]
async fn pending_and_active_visibility_through_current_model() {
    let li = LearnedIndex::new();
    assert_eq!(li.current_model().await.version, 1, "no pending → active");

    let v5 = RmiModel {
        version: 5,
        ..RmiModel::default()
    };
    li.set_pending_model(v5.clone()).await;
    assert_eq!(
        li.current_model().await.version,
        5,
        "pending present → returned"
    );
    let (ready, has_update) = li.status().await;
    assert!(has_update);
    assert!(!ready);

    li.activate(RmiModel::default()).await;
    let (ready, has_update) = li.status().await;
    assert!(!has_update, "activate clears pending");
    assert!(!ready, "activate clears update_ready");
    assert_eq!(li.version().await, 1);
    assert_eq!(li.next_version(), 2);
}

#[tokio::test]
async fn reset_drift_clears_update_ready() {
    let li = LearnedIndex::new();
    for _ in 0..50 {
        let _ = li
            .record_insert("k", 0, 50, 0.40, PID_ADJUST_INTERVAL)
            .await;
    }
    assert!(li.is_update_ready().await);
    li.reset_drift(999).await;
    assert!(!li.is_update_ready().await);
}

#[tokio::test]
async fn record_insert_returns_pid_due_at_interval() {
    let li = LearnedIndex::new();
    for i in 0..PID_ADJUST_INTERVAL {
        assert!(
            !li.record_insert("k", 10_000, 50, 0.40, PID_ADJUST_INTERVAL)
                .await,
            "call {i} should not trigger PID"
        );
    }
    assert!(
        li.record_insert("k", 10_000, 50, 0.40, PID_ADJUST_INTERVAL)
            .await,
        "call {} should trigger PID",
        PID_ADJUST_INTERVAL
    );
}

#[test]
fn next_version_starts_at_two() {
    let li = LearnedIndex::new();
    assert_eq!(li.next_version(), 2);
}
