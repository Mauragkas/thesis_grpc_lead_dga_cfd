//! Tests for the concrete `InMemoryGeneStore`, distance metrics, eviction
//! policies, and the `GeneRecord` data shape.

use orchestrator::config::GeneStoreConfig;
use orchestrator::gene_store::record::GeneRecord;
use orchestrator::gene_store::GeneStore;
use orchestrator::gene_store::{
    DistanceMetric, EuclideanDistance, EvictionPolicy, GenerationEvictor, InMemoryGeneStore,
};

// --- Distance metric tests (OCP: trait-based) ---

#[test]
fn euclidean_distance_zero_for_identical_vectors() {
    let m = EuclideanDistance;
    assert!((m.distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])).abs() < 1e-12);
}

#[test]
fn euclidean_distance_known_value() {
    let m = EuclideanDistance;
    // (3-0)^2 + (4-0)^2 = 25 => sqrt = 5
    assert!((m.distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-12);
}

#[test]
fn euclidean_distance_handles_empty_vectors() {
    let m = EuclideanDistance;
    assert!((m.distance(&[], &[])).abs() < 1e-12);
}

// --- Eviction policy tests (OCP: trait-based) ---

#[test]
fn generation_evictor_keeps_fresh_records() {
    let p = GenerationEvictor { max_age: 5 };
    let rec = GeneRecord {
        genes: vec![0.5],
        fitness: 1.0,
        generation: 1,
        last_accessed_gen: 8,
    };
    assert!(!p.should_evict(&rec, 10));
}

#[test]
fn generation_evictor_evicts_stale_records() {
    let p = GenerationEvictor { max_age: 5 };
    let rec = GeneRecord {
        genes: vec![0.5],
        fitness: 1.0,
        generation: 1,
        last_accessed_gen: 1,
    };
    assert!(p.should_evict(&rec, 10));
}

#[test]
fn generation_evictor_boundary_is_inclusive() {
    let p = GenerationEvictor { max_age: 5 };
    let rec = GeneRecord {
        genes: vec![0.5],
        fitness: 1.0,
        generation: 1,
        last_accessed_gen: 5,
    };
    // 10 - 5 = 5, not strictly greater than 5 => keep
    assert!(!p.should_evict(&rec, 10));
    // 11 - 5 = 6 > 5 => evict
    assert!(p.should_evict(&rec, 11));
}

#[test]
fn custom_eviction_policy_can_be_substituted() {
    // OCP: a brand-new policy without touching the store.
    struct AlwaysEvict;
    impl EvictionPolicy for AlwaysEvict {
        fn should_evict(&self, _r: &GeneRecord, _gen: usize) -> bool {
            true
        }
    }
    let rec = GeneRecord {
        genes: vec![0.5],
        fitness: 1.0,
        generation: 1,
        last_accessed_gen: 1,
    };
    assert!(AlwaysEvict.should_evict(&rec, 1));
}

// --- InMemoryGeneStore tests ---

fn make_store() -> InMemoryGeneStore<EuclideanDistance, GenerationEvictor> {
    InMemoryGeneStore::new(
        EuclideanDistance,
        GenerationEvictor { max_age: 5 },
    )
}

#[tokio::test]
async fn store_and_lookup_exact_roundtrip() {
    let s = make_store();
    let genes = vec![0.1, 0.2, 0.3];
    s.store(genes.clone(), 0.9, 1).await;
    assert_eq!(s.lookup_exact(&genes, 2).await, Some(0.9));
}

#[tokio::test]
async fn lookup_exact_returns_none_for_unknown() {
    let s = make_store();
    assert_eq!(s.lookup_exact(&[0.0, 0.0], 1).await, None);
}

#[tokio::test]
async fn lookup_exact_refreshes_last_accessed_generation() {
    // OCP/LSP: behaviour contract — access refreshes TTL.
    let s = make_store();
    let genes = vec![0.5];
    s.store(genes.clone(), 1.0, 1).await;
    // Access at generation 7 (within max_age=5 of gen 7) refreshes.
    s.lookup_exact(&genes, 7).await;
    // Evict at generation 8: last_accessed=7 => 8-7=1 <= 5, retained.
    s.evict_expired(8).await;
    assert_eq!(s.lookup_exact(&genes, 9).await, Some(1.0));
}

#[tokio::test]
async fn query_knn_returns_closest_first() {
    let s = make_store();
    s.store(vec![0.0], 1.0, 1).await;
    s.store(vec![1.0], 2.0, 1).await;
    s.store(vec![0.5], 3.0, 1).await;

    let res = s.query_knn(&[0.4], 2, 1).await;
    assert_eq!(res.len(), 2);
    // Closest to 0.4 is 0.5 (fitness 3.0), then 0.0 (fitness 1.0)
    assert!((res[0].genes[0] - 0.5).abs() < 1e-12);
    assert!((res[1].genes[0] - 0.0).abs() < 1e-12);
}

#[tokio::test]
async fn query_knn_handles_empty_store() {
    let s = make_store();
    let res = s.query_knn(&[0.0], 3, 1).await;
    assert!(res.is_empty());
}

#[tokio::test]
async fn query_knn_k_larger_than_population_returns_all() {
    let s = make_store();
    s.store(vec![0.0], 1.0, 1).await;
    let res = s.query_knn(&[0.0], 99, 1).await;
    assert_eq!(res.len(), 1);
}

#[tokio::test]
async fn evict_expired_drops_stale_records() {
    let s = make_store();
    s.store(vec![0.0], 1.0, 1).await; // last_accessed=1
    s.store(vec![1.0], 2.0, 1).await;
    // Touch the second record late so it survives.
    s.lookup_exact(&[1.0], 6).await; // last_accessed=6
                                     // gen 7: rec1 => 7-1=6 > 5 evict; rec2 => 7-6=1 <= 5 keep
    s.evict_expired(7).await;
    assert_eq!(s.lookup_exact(&[0.0], 8).await, None);
    assert_eq!(s.lookup_exact(&[1.0], 8).await, Some(2.0));
}

// --- GeneStoreConfig sanity ---

#[test]
fn gene_store_config_default_matches_evictor_default() {
    let cfg = GeneStoreConfig::default();
    let ev = GenerationEvictor::default();
    assert_eq!(cfg.max_age_generations, ev.max_age);
}
