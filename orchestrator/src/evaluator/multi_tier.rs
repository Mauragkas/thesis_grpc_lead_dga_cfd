use crate::config::TierConfig;
use crate::evaluator::tier_metrics::TierMetricsTracker;
use crate::evaluator::r#trait::Evaluator;
use crate::gene_store::metric::DistanceMetric;
use crate::gene_store::metric::EuclideanDistance;
use crate::gene_store::GeneStore;
use crate::neighbor_store::NeighborStore;
use crate::surrogate_client::SurrogateClient;
use std::sync::Arc;
use tonic::Status;
use tracing::{debug, info, warn};

/// Multi-Tier (ε-Bypass) Evaluator.
///
/// Implements a hierarchical evaluation strategy:
/// - Tier 1: Cache Hit (d_min < ε_exact) -> Return closest neighbor fitness directly (0 FLOPs).
/// - Tier 2: Surrogate Interpolation (ε_exact <= d_min <= R) -> Predict via external MLP surrogate service.
/// - Tier 3: Extrapolation / Void Region (d_min > R or surrogate not ready) -> True expensive worker simulation,
///   followed by active learning feedback into the LEAD DHT and surrogate sliding window buffer.
pub struct MultiTierEvaluator {
    simulator: Arc<dyn Evaluator>,
    store: Arc<dyn GeneStore>,
    neighbor_store: Option<Arc<dyn NeighborStore>>,
    surrogate: Option<Arc<dyn SurrogateClient>>,
    config: TierConfig,
    metric: EuclideanDistance,
    pub metrics: Arc<TierMetricsTracker>,
}

impl MultiTierEvaluator {
    pub fn new(
        simulator: Arc<dyn Evaluator>,
        store: Arc<dyn GeneStore>,
        neighbor_store: Option<Arc<dyn NeighborStore>>,
        surrogate: Option<Arc<dyn SurrogateClient>>,
        config: TierConfig,
    ) -> Self {
        Self {
            simulator,
            store,
            neighbor_store,
            surrogate,
            config,
            metric: EuclideanDistance,
            metrics: Arc::new(TierMetricsTracker::new()),
        }
    }

    /// Finds the closest known neighbor and its distance.
    async fn find_closest_neighbor(
        &self,
        query: &[f64],
    ) -> Option<(Vec<f64>, f64, f64)> {
        // 1. First check local in-memory store
        let local_knn = self.store.query_knn(query, 1, 0).await;
        if let Some(first) = local_knn.first() {
            let dist = self.metric.distance(query, &first.genes);
            return Some((first.genes.clone(), first.fitness, dist));
        }

        // 2. If neighbor_store (LEAD DHT) is available, check distributed index
        if let Some(ns) = &self.neighbor_store {
            if let Ok(dht_knn) = ns.query_knn(query, 1).await {
                if let Some(first) = dht_knn.first() {
                    let dist = self.metric.distance(query, &first.genes);
                    return Some((first.genes.clone(), first.fitness, dist));
                }
            }
        }

        None
    }
}

#[async_trait::async_trait]
impl Evaluator for MultiTierEvaluator {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
        let n = population.len();
        let mut fitnesses = vec![f64::NEG_INFINITY; n];

        let mut tier1_hits = 0;
        let mut tier2_indices = Vec::new();
        let mut tier3_indices = Vec::new();

        // ── Phase 1: Spatial Proximity & Tier Partitioning ───────────────── //
        for (i, ind) in population.iter().enumerate() {
            // First check exact match in local store
            if let Some(exact_fit) = self.store.lookup_exact(ind, 0).await {
                fitnesses[i] = exact_fit;
                tier1_hits += 1;
                continue;
            }

            // Spatial proximity query
            match self.find_closest_neighbor(ind).await {
                Some((_nn_genes, nn_fit, d_min)) => {
                    if d_min < self.config.epsilon_exact {
                        // Tier 1: ε-Bypass (Cache Hit)
                        fitnesses[i] = nn_fit;
                        tier1_hits += 1;
                        debug!("Ind {i}: Tier 1 Cache Hit (d_min={d_min:.6} < ε={:.6})", self.config.epsilon_exact);
                    } else if d_min <= self.config.radius_r {
                        // Tier 2 candidate: within interpolation radius
                        tier2_indices.push(i);
                        debug!("Ind {i}: Tier 2 Candidate (ε <= d_min={d_min:.4} <= R={:.4})", self.config.radius_r);
                    } else {
                        // Tier 3: void region / unexplored
                        tier3_indices.push(i);
                        debug!("Ind {i}: Tier 3 Void Region (d_min={d_min:.4} > R={:.4})", self.config.radius_r);
                    }
                }
                None => {
                    // Cold start: no known neighbors yet
                    tier3_indices.push(i);
                }
            }
        }

        if tier1_hits > 0 {
            self.metrics.record_tier1_hit(tier1_hits);
        }

        // ── Phase 2: Tier 2 Surrogate Evaluation & Fallback ──────────────── //
        if !tier2_indices.is_empty() {
            let mut surrogate_success = false;

            if let Some(surr) = &self.surrogate {
                let tier2_inds: Vec<Vec<f64>> = tier2_indices.iter().map(|&idx| population[idx].clone()).collect();
                match surr.predict_batch(&tier2_inds).await {
                    Ok(Some(preds)) if preds.len() == tier2_indices.len() => {
                        for (&idx, fit) in tier2_indices.iter().zip(preds.into_iter()) {
                            fitnesses[idx] = fit;
                        }
                        self.metrics.record_tier2_hit(tier2_indices.len());
                        info!(
                            "Tier 2: evaluated {} individuals via MLP Surrogate",
                            tier2_indices.len()
                        );
                        surrogate_success = true;
                    }
                    Ok(Some(_)) => {
                        warn!("Surrogate returned unexpected batch length; falling back to Tier 3");
                    }
                    Ok(None) => {
                        debug!("Surrogate reported not ready; routing {} Tier 2 individuals to Tier 3", tier2_indices.len());
                    }
                    Err(e) => {
                        warn!("Surrogate error: {e}; falling back to Tier 3");
                    }
                }
            }

            if !surrogate_success {
                // Fallback to Tier 3
                tier3_indices.extend_from_slice(&tier2_indices);
            }
        }

        // ── Phase 3: Tier 3 True Worker Simulation & Feedback Loop ──────── //
        if !tier3_indices.is_empty() {
            let tier3_inds: Vec<Vec<f64>> = tier3_indices.iter().map(|&idx| population[idx].clone()).collect();
            info!(
                "Tier 3: evaluating {} individuals via True Worker Simulator",
                tier3_inds.len()
            );

            let fresh_fits = self.simulator.evaluate_population(&tier3_inds).await?;
            self.metrics.record_tier3_eval(tier3_inds.len());

            let mut feedback_samples = Vec::with_capacity(tier3_inds.len());

            for (&idx, (genes, &fit)) in tier3_indices.iter().zip(tier3_inds.iter().zip(fresh_fits.iter())) {
                fitnesses[idx] = fit;
                // Store in local store
                self.store.store(genes.clone(), fit, 0).await;

                // Store in distributed LEAD DHT
                if let Some(ns) = &self.neighbor_store {
                    if let Err(e) = ns.store(genes, fit, 0).await {
                        warn!("Failed to store in LEAD neighbor store: {e}");
                    }
                }

                // Collect for surrogate ingestion (filter out invalid/rejected geometries)
                if fit > -1e8 {
                    feedback_samples.push((genes.clone(), fit));
                }
            }

            // Ingest into surrogate sliding window
            if let Some(surr) = &self.surrogate {
                if !feedback_samples.is_empty() {
                    let surr_clone = surr.clone();
                    tokio::spawn(async move {
                        if let Err(e) = surr_clone.ingest_samples(&feedback_samples).await {
                            warn!("Failed to ingest feedback samples into surrogate: {e}");
                        }
                    });
                }
            }
        }

        let snap = self.metrics.snapshot();
        info!(
            "Multi-Tier Eval Stats: Total={}, T1={:.1}% ({} hits), T2={:.1}% ({} surr), T3={:.1}% ({} sims) | Bypass={:.1}%",
            snap.total_evaluations,
            snap.tier1_ratio() * 100.0,
            snap.tier1_exact_hits,
            snap.tier2_ratio() * 100.0,
            snap.tier2_surrogate_hits,
            snap.tier3_ratio() * 100.0,
            snap.tier3_simulator_evals,
            snap.bypass_ratio() * 100.0
        );

        Ok(fitnesses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene_store::eviction::GenerationEvictor;
    use crate::gene_store::in_memory::InMemoryGeneStore;
    use crate::surrogate_client::mock::MockSurrogateClient;
    use tonic::Status;

    struct MockSimulator {
        fitness: f64,
    }

    #[async_trait::async_trait]
    impl Evaluator for MockSimulator {
        async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
            Ok(vec![self.fitness; population.len()])
        }
    }

    #[tokio::test]
    async fn test_tier1_cache_hit_exact() {
        let sim = Arc::new(MockSimulator { fitness: 100.0 });
        let store = Arc::new(InMemoryGeneStore::new(
            EuclideanDistance,
            GenerationEvictor { max_age: 10 },
        ));
        let surrogate = Arc::new(MockSurrogateClient::new(true, 50.0));

        let config = TierConfig {
            epsilon_exact: 0.01,
            radius_r: 0.20,
            k_neighbors: 5,
            min_neighbors: 1,
        };

        let evaluator = MultiTierEvaluator::new(sim, store.clone(), None, Some(surrogate), config);

        // Pre-populate store with a known design
        store.store(vec![0.5, 0.5], 42.0, 0).await;

        // Query with identical design -> should be Tier 1 hit (42.0)
        let pop = vec![vec![0.5, 0.5]];
        let fits = evaluator.evaluate_population(&pop).await.unwrap();

        assert_eq!(fits[0], 42.0);
        let snap = evaluator.metrics.snapshot();
        assert_eq!(snap.tier1_exact_hits, 1);
        assert_eq!(snap.tier2_surrogate_hits, 0);
        assert_eq!(snap.tier3_simulator_evals, 0);
    }

    #[tokio::test]
    async fn test_tier2_surrogate_interpolation() {
        let sim = Arc::new(MockSimulator { fitness: 100.0 });
        let store = Arc::new(InMemoryGeneStore::new(
            EuclideanDistance,
            GenerationEvictor { max_age: 10 },
        ));
        let surrogate = Arc::new(MockSurrogateClient::new(true, 77.0));

        let config = TierConfig {
            epsilon_exact: 0.01,
            radius_r: 0.20,
            k_neighbors: 5,
            min_neighbors: 1,
        };

        let evaluator = MultiTierEvaluator::new(sim, store.clone(), None, Some(surrogate), config);

        // Pre-populate store with a point at distance ~0.05
        store.store(vec![0.5, 0.5], 42.0, 0).await;

        // Query with design at distance 0.05 (between ε=0.01 and R=0.20)
        let pop = vec![vec![0.54, 0.53]];
        let fits = evaluator.evaluate_population(&pop).await.unwrap();

        assert_eq!(fits[0], 77.0); // Surrogate predicted 77.0
        let snap = evaluator.metrics.snapshot();
        assert_eq!(snap.tier1_exact_hits, 0);
        assert_eq!(snap.tier2_surrogate_hits, 1);
        assert_eq!(snap.tier3_simulator_evals, 0);
    }

    #[tokio::test]
    async fn test_tier3_void_region_and_feedback() {
        let sim = Arc::new(MockSimulator { fitness: 99.0 });
        let store = Arc::new(InMemoryGeneStore::new(
            EuclideanDistance,
            GenerationEvictor { max_age: 10 },
        ));
        let surrogate = Arc::new(MockSurrogateClient::new(true, 50.0));

        let config = TierConfig {
            epsilon_exact: 0.01,
            radius_r: 0.10,
            k_neighbors: 5,
            min_neighbors: 1,
        };

        let evaluator = MultiTierEvaluator::new(sim, store.clone(), None, Some(surrogate.clone()), config);

        // Pre-populate store with a point at distance ~0.70 (far away > R=0.10)
        store.store(vec![0.0, 0.0], 10.0, 0).await;

        let pop = vec![vec![0.5, 0.5]];
        let fits = evaluator.evaluate_population(&pop).await.unwrap();

        assert_eq!(fits[0], 99.0); // Simulator returned 99.0
        let snap = evaluator.metrics.snapshot();
        assert_eq!(snap.tier1_exact_hits, 0);
        assert_eq!(snap.tier2_surrogate_hits, 0);
        assert_eq!(snap.tier3_simulator_evals, 1);
    }
}
