use std::collections::HashMap;
use tempfile::TempDir;

use lead_node::config::{Config, StorageBackend};
use lead_node::storage::{InMemoryStore, KeyStore, SledStore, StorageEngine};

#[tokio::test]
async fn test_in_memory_store_operations() {
    let store = InMemoryStore::new();
    assert!(store.is_empty().await);
    assert_eq!(store.len().await, 0);

    store.put("k1".into(), "v1".into()).await;
    store.put("k2".into(), "v2".into()).await;
    store.put("k3".into(), "v3".into()).await;

    assert_eq!(store.len().await, 3);
    assert!(!store.is_empty().await);
    assert_eq!(store.get("k1").await, Some("v1".into()));
    assert_eq!(store.get("unknown").await, None);

    let range = store.range_scan("k1", 2).await;
    assert_eq!(range, vec![("k1".into(), "v1".into()), ("k2".into(), "v2".into())]);

    let after = store.range_scan_after("k1", 2).await;
    assert_eq!(after, vec![("k2".into(), "v2".into()), ("k3".into(), "v3".into())]);

    let removed = store.remove("k2").await;
    assert_eq!(removed, Some("v2".into()));
    assert_eq!(store.len().await, 2);
    assert_eq!(store.get("k2").await, None);
}

#[tokio::test]
async fn test_sled_store_basic_operations() {
    let store = SledStore::open_temporary().expect("failed to open temporary sled store");
    assert!(store.is_empty().await);
    assert_eq!(store.len().await, 0);

    store.put("alpha".into(), "val_a".into()).await;
    store.put("beta".into(), "val_b".into()).await;
    store.put("gamma".into(), "val_c".into()).await;

    assert_eq!(store.len().await, 3);
    assert!(!store.is_empty().await);
    assert_eq!(store.get("beta").await, Some("val_b".into()));
    assert_eq!(store.get("nonexistent").await, None);

    let range = store.range_scan("alpha", 2).await;
    assert_eq!(
        range,
        vec![("alpha".into(), "val_a".into()), ("beta".into(), "val_b".into())]
    );

    let after = store.range_scan_after("alpha", 2).await;
    assert_eq!(
        after,
        vec![("beta".into(), "val_b".into()), ("gamma".into(), "val_c".into())]
    );

    let removed = store.remove("beta").await;
    assert_eq!(removed, Some("val_b".into()));
    assert_eq!(store.len().await, 2);
    assert_eq!(store.get("beta").await, None);
}

#[tokio::test]
async fn test_sled_store_durability_across_restarts() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let path = temp_dir.path().join("lead_db");

    // Phase 1: Open store, write candidate evaluations, flush and drop
    {
        let store = SledStore::open(&path).expect("failed to open initial sled store");
        store.put("cand_001".into(), r#"{"fitness": 123.45}"#.into()).await;
        store.put("cand_002".into(), r#"{"fitness": 234.56}"#.into()).await;
        store.put("cand_003".into(), r#"{"fitness": 345.67}"#.into()).await;
        store.flush().await;
        assert_eq!(store.len().await, 3);
        // store is dropped here, closing the sled DB
    }

    // Phase 2: Simulate node restart — reopen database from same directory
    {
        let restored_store = SledStore::open(&path).expect("failed to reopen sled store");
        assert_eq!(restored_store.len().await, 3);
        assert!(!restored_store.is_empty().await);

        assert_eq!(
            restored_store.get("cand_001").await,
            Some(r#"{"fitness": 123.45}"#.into())
        );
        assert_eq!(
            restored_store.get("cand_002").await,
            Some(r#"{"fitness": 234.56}"#.into())
        );
        assert_eq!(
            restored_store.get("cand_003").await,
            Some(r#"{"fitness": 345.67}"#.into())
        );

        let range = restored_store.range_scan("cand_001", 10).await;
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].0, "cand_001");
        assert_eq!(range[1].0, "cand_002");
        assert_eq!(range[2].0, "cand_003");

        // Mutate in second session and verify again
        restored_store.remove("cand_002").await;
        restored_store.put("cand_004".into(), r#"{"fitness": 456.78}"#.into()).await;
        restored_store.flush().await;
    }

    // Phase 3: Simulate second restart — verify mutations persisted
    {
        let third_store = SledStore::open(&path).expect("failed to reopen sled store for third check");
        assert_eq!(third_store.len().await, 3);
        assert_eq!(third_store.get("cand_002").await, None);
        assert_eq!(
            third_store.get("cand_004").await,
            Some(r#"{"fitness": 456.78}"#.into())
        );
    }
}

#[tokio::test]
async fn test_storage_engine_dispatch() {
    let mem_engine = StorageEngine::memory();
    mem_engine.put("k1".into(), "v1".into()).await;
    assert_eq!(mem_engine.get("k1").await, Some("v1".into()));
    assert_eq!(mem_engine.len().await, 1);

    let sled_engine = StorageEngine::sled_temporary().expect("temporary sled engine");
    sled_engine.put("k2".into(), "v2".into()).await;
    assert_eq!(sled_engine.get("k2").await, Some("v2".into()));
    assert_eq!(sled_engine.len().await, 1);
}

#[test]
fn test_config_storage_parsing() {
    // Default: memory backend, no path
    let default_cfg = Config::default();
    assert_eq!(default_cfg.storage_backend, StorageBackend::Memory);
    assert_eq!(default_cfg.storage_path, None);

    // LEAD_STORAGE_PATH sets sled backend automatically
    let mut vars = HashMap::new();
    vars.insert("LEAD_STORAGE_PATH", "/custom/data/lead");
    let cfg = Config::from_vars(vars);
    assert_eq!(cfg.storage_backend, StorageBackend::Sled);
    assert_eq!(cfg.storage_path.as_deref(), Some("/custom/data/lead"));

    // Explicit LEAD_STORAGE_BACKEND=memory overrides path
    let mut vars = HashMap::new();
    vars.insert("LEAD_STORAGE_PATH", "/custom/data/lead");
    vars.insert("LEAD_STORAGE_BACKEND", "memory");
    let cfg = Config::from_vars(vars);
    assert_eq!(cfg.storage_backend, StorageBackend::Memory);

    // Explicit LEAD_STORAGE_BACKEND=sled without path defaults path
    let mut vars = HashMap::new();
    vars.insert("LEAD_STORAGE_BACKEND", "sled");
    let cfg = Config::from_vars(vars);
    assert_eq!(cfg.storage_backend, StorageBackend::Sled);
    assert_eq!(cfg.storage_path.as_deref(), Some("./data/lead"));
}
