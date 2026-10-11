//! Tests for the Hilbert embedding + key generator.

use orchestrator::hilbert::{GeneKeyGenerator, HilbertKeyGenerator, NUM_CURVES};

fn leading_u64(key: &str) -> u64 {
    let hex = key.split('|').next().unwrap();
    let prefix = &hex[..hex.len().min(16)];
    u64::from_str_radix(prefix, 16).unwrap()
}

fn hex_dist(a: &str, b: &str) -> u128 {
    (leading_u64(a) as i128 - leading_u64(b) as i128).unsigned_abs()
}

fn euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn keys_are_deterministic() {
    let kg = HilbertKeyGenerator::new(10);
    let genes = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25];
    assert_eq!(kg.keys_for(&genes), kg.keys_for(&genes));
}

#[test]
fn produces_one_key_per_curve_with_distinct_curve_ids() {
    let kg = HilbertKeyGenerator::new(10);
    let genes = [0.5; 10];
    let keys = kg.keys_for(&genes);
    assert_eq!(keys.len(), NUM_CURVES);
    let mut curves: Vec<char> = keys.iter().map(|k| k.chars().next().unwrap()).collect();
    curves.sort();
    curves.dedup();
    assert_eq!(curves.len(), NUM_CURVES);
}

#[test]
fn key_roundtrips_through_parse() {
    let kg = HilbertKeyGenerator::new(10);
    let genes = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25];
    for key in kg.keys_for(&genes) {
        assert_eq!(kg.parse_key(&key), Some(genes.clone()));
    }
}

#[test]
fn multi_probe_recovers_near_neighbor() {
    let kg = HilbertKeyGenerator::new(10);
    let a = vec![0.5; 10];
    let b = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51];
    let far = vec![0.0; 10];

    // Sanity: b is genuinely closer to a than far is.
    assert!(euclid(&a, &b) < euclid(&a, &far));

    let keys_a = kg.keys_for(&a);
    let keys_b = kg.keys_for(&b);
    let keys_far = kg.keys_for(&far);

    // Multi-probe guarantee: the closest curve for the near neighbor is
    // closer than the closest curve for the far point.
    let min_ab = (0..NUM_CURVES)
        .map(|i| hex_dist(&keys_a[i], &keys_b[i]))
        .min()
        .unwrap();
    let min_af = (0..NUM_CURVES)
        .map(|i| hex_dist(&keys_a[i], &keys_far[i]))
        .min()
        .unwrap();
    assert!(
        min_ab < min_af,
        "multi-probe failed to recover near neighbor"
    );
}

#[derive(Default, Clone)]
struct BTreeLeadStore {
    data: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, String>>>,
}

impl BTreeLeadStore {
    fn key_count(&self) -> usize {
        self.data.lock().unwrap().len()
    }
}

#[tonic::async_trait]
impl orchestrator::lead_store::LeadStore for BTreeLeadStore {
    async fn store_gene(&self, key: &str, value: &str) -> Result<(), tonic::Status> {
        self.data
            .lock()
            .unwrap()
            .insert(key.to_string(), value.to_string());
        Ok(())
    }

    async fn get_gene(&self, key: &str) -> Result<Option<String>, tonic::Status> {
        Ok(self.data.lock().unwrap().get(key).cloned())
    }

    async fn range_query(
        &self,
        start_key: &str,
        count: u64,
    ) -> Result<Vec<(String, String)>, tonic::Status> {
        Ok(self
            .data
            .lock()
            .unwrap()
            .range(start_key.to_string()..)
            .take(count as usize)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }
}

#[tokio::test]
async fn curve_count_sweep_improves_knn_recall_and_scales_storage_linearly() {
    use orchestrator::neighbor_store::{HilbertNeighborStore, NeighborStore};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let n_points = 80;
    let ndims = 10;
    let k = 10;
    let mut rng = StdRng::seed_from_u64(42);
    let points: Vec<Vec<f64>> = (0..n_points)
        .map(|_| (0..ndims).map(|_| rng.gen::<f64>()).collect())
        .collect();

    let mut mean_dists = Vec::new();
    for &c in &[1usize, 2, 3, 4, 6, 8] {
        let store = BTreeLeadStore::default();
        let kg = HilbertKeyGenerator::with_curves(ndims, c);
        let ns = HilbertNeighborStore::new(kg, store.clone());

        for (i, pt) in points.iter().enumerate() {
            ns.store(pt, i as f64, 0).await.unwrap();
        }
        assert_eq!(store.key_count(), n_points * c);

        // Query slightly perturbed versions of the first 20 points so exact self-key is skipped
        let mut total_dist = 0.0;
        let n_queries = 20;
        for pt in points.iter().take(n_queries) {
            let mut q = pt.clone();
            q[0] = (q[0] - 0.01).clamp(0.0, 1.0);
            let neighbors = ns.query_knn(&q, k).await.unwrap();
            let avg_d: f64 = neighbors
                .iter()
                .map(|p| euclid(&q, &p.genes))
                .sum::<f64>()
                / (neighbors.len().max(1) as f64);
            total_dist += avg_d;
        }
        mean_dists.push((c, total_dist / n_queries as f64));
    }

    // More curves must yield closer retrieved neighbors (lower mean Euclidean distance) than 1 curve
    let (_, dist_c1) = mean_dists[0];
    let (_, dist_c8) = *mean_dists.last().unwrap();
    assert!(
        dist_c8 < dist_c1,
        "Expected 8 curves ({dist_c8:.4}) to retrieve closer neighbors than 1 curve ({dist_c1:.4})"
    );
}

