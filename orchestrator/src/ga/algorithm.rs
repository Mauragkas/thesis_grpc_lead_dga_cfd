use crate::config::GaConfig;
use crate::evaluator::Evaluator;
use crate::ga::operators::{next_generation, random_population, select_survivors};
use crate::ga::telemetry::{
    compute_fitness_stats, compute_gene_variance, compute_population_entropy, GenerationRecord,
    TelemetrySink,
};
use crate::gene_store::GeneStore;
use crate::migration::MigrationHook;
use crate::neighbor_store::NeighborStore;
use rand::rngs::StdRng;
use rand_distr::Normal;
use std::time::Instant;
use tonic::Status;
use tracing::{error, info, warn};


/// Result of a completed GA run containing the best candidate found.
#[derive(Debug, Clone, PartialEq)]
pub struct GaResult {
    /// Best fitness achieved across all evaluated generations.
    pub best_fitness: f64,
    /// Genome (normalized gene vector) of the best candidate.
    pub best_genome: Vec<f64>,
}

impl std::ops::Deref for GaResult {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.best_fitness
    }
}

/// Finds the best individual and its fitness from a population and corresponding fitnesses.
pub fn find_best_candidate<'b>(
    population: &'b [Vec<f64>],
    fitnesses: &[f64],
) -> Option<(&'b [f64], f64)> {
    fitnesses
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .and_then(|(idx, &fitness)| {
            population.get(idx).map(|ind| (ind.as_slice(), fitness))
        })
}

/// Tracks generational progress and detects stagnation/convergence.
/// SRP: encapsulates convergence detection without mixing with evaluation or migration.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    patience: usize,
    min_improvement: f64,
    min_generations: usize,
    last_improvement_fitness: f64,
    stagnant_generations: usize,
}

impl ProgressTracker {
    pub fn new(patience: usize, min_improvement: f64, min_generations: usize) -> Self {
        Self {
            patience,
            min_improvement,
            min_generations,
            last_improvement_fitness: f64::NEG_INFINITY,
            stagnant_generations: 0,
        }
    }

    /// Records the current best fitness and generation index.
    /// Returns `true` if early stopping condition is met (stagnated for >= patience generations
    /// and current_gen >= min_generations).
    pub fn update(&mut self, current_best: f64, current_gen: usize) -> bool {
        if self.patience == 0 {
            // Patience of 0 disables early stopping.
            return false;
        }

        if current_best > self.last_improvement_fitness + self.min_improvement {
            self.last_improvement_fitness = current_best;
            self.stagnant_generations = 0;
            false
        } else {
            self.stagnant_generations += 1;
            self.is_stagnant(current_gen)
        }
    }

    pub fn stagnant_generations(&self) -> usize {
        self.stagnant_generations
    }

    pub fn last_improvement_fitness(&self) -> f64 {
        self.last_improvement_fitness
    }

    pub fn is_stagnant(&self, current_gen: usize) -> bool {
        self.patience > 0
            && self.stagnant_generations >= self.patience
            && current_gen >= self.min_generations
    }
}

/// SRP: orchestrates the generational loop. Depends on the `Evaluator`,
/// `GeneStore`, `NeighborStore`, and `MigrationHook` abstractions (DIP), never on concrete
/// impls directly.
pub struct GaRunner<'a> {
    pub cfg: &'a GaConfig,
    pub evaluator: &'a dyn Evaluator,
    pub store: &'a dyn GeneStore,
    pub neighbor_store: Option<&'a dyn NeighborStore>,
    pub migration: Option<&'a dyn MigrationHook>,
    pub telemetry: Option<&'a dyn TelemetrySink>,
}

impl<'a> GaRunner<'a> {
    pub fn new(
        cfg: &'a GaConfig,
        evaluator: &'a dyn Evaluator,
        store: &'a dyn GeneStore,
    ) -> Self {
        Self {
            cfg,
            evaluator,
            store,
            neighbor_store: None,
            migration: None,
            telemetry: None,
        }
    }

    pub fn with_neighbor_store(mut self, ns: Option<&'a dyn NeighborStore>) -> Self {
        self.neighbor_store = ns;
        self
    }

    pub fn with_migration(mut self, mig: Option<&'a dyn MigrationHook>) -> Self {
        self.migration = mig;
        self
    }

    pub fn with_telemetry(mut self, telem: Option<&'a dyn TelemetrySink>) -> Self {
        self.telemetry = telem;
        self
    }

    pub async fn run(&self, rng: &mut StdRng) -> Result<GaResult, Status> {

        let start = Instant::now();
        info!(
            "GA run starting: pop_size={}, genes_len={}, max_generations={}, min_generations={}, stagnation_patience={}, min_improvement={}, mut_sigma={}, elite_frac={}, batch_size={}",
            self.cfg.pop_size,
            self.cfg.genes_len,
            self.cfg.max_generations,
            self.cfg.min_generations,
            self.cfg.stagnation_patience,
            self.cfg.min_improvement,
            self.cfg.mut_sigma,
            self.cfg.elite_frac,
            self.cfg.batch_size
        );
        let mut population = random_population(rng, self.cfg);
        let mut best_ever = f64::NEG_INFINITY;
        let mut best_genome = Vec::new();
        let normal = Normal::new(0.0, self.cfg.mut_sigma).unwrap();
        let mut tracker = ProgressTracker::new(
            self.cfg.stagnation_patience,
            self.cfg.min_improvement,
            self.cfg.min_generations,
        );
        let mut prev_tier_snapshot: Option<crate::evaluator::tier_metrics::TierMetricsSnapshot> = None;

        for gen in 1..=self.cfg.max_generations {
            info!("Evaluating generation {gen}/{}...", self.cfg.max_generations);

            self.integrate_immigrants(&mut population, gen).await;

            let fitnesses = self.evaluate_generation(&population, gen).await?;

            self.store.evict_expired(gen).await;

            let curr_tier_snapshot = self.evaluator.tier_metrics();
            self.record_generation_stats(
                gen,
                start.elapsed().as_secs_f64(),
                &population,
                &fitnesses,
                &mut best_ever,
                &mut best_genome,
                curr_tier_snapshot,
                prev_tier_snapshot,
            )
            .await;
            prev_tier_snapshot = curr_tier_snapshot;


            self.handle_emigration(gen, &population, &fitnesses).await;

            if tracker.update(best_ever, gen) {
                info!(
                    gen = gen,
                    best_fitness = best_ever,
                    stagnant_generations = tracker.stagnant_generations(),
                    "GA converged: no significant progress (> {}) for {} consecutive generations. Stopping at generation {gen}/{}.",
                    self.cfg.min_improvement,
                    tracker.stagnant_generations(),
                    self.cfg.max_generations
                );
                break;
            }

            population = self.advance_population(&population, &fitnesses, rng, &normal);
        }

        let elapsed = start.elapsed().as_secs_f64();
        info!(
            elapsed = elapsed,
            best_fitness = best_ever,
            best_genome = ?best_genome,
            "Finished GA run in {elapsed:.2}s. Best fitness: {best_ever:.4}; best genome: {best_genome:?}"
        );
        Ok(GaResult {
            best_fitness: best_ever,
            best_genome,
        })
    }

    /// Drains incoming immigrants and replaces the tail of the population.
    async fn integrate_immigrants(&self, population: &mut Vec<Vec<f64>>, gen: usize) {
        if let Some(mig) = self.migration {
            let immigrants = mig.drain_immigrants().await;
            if !immigrants.is_empty() {
                info!(
                    "Gen {gen}: integrating {} immigrants into population",
                    immigrants.len()
                );
                for im in immigrants {
                    population.push(im.genes);
                }
                population.truncate(self.cfg.pop_size);
            }
        }
    }

    /// Evaluates population members via the injected Evaluator abstraction.
    async fn evaluate_generation(
        &self,
        population: &[Vec<f64>],
        gen: usize,
    ) -> Result<Vec<f64>, Status> {
        self.evaluator
            .evaluate_population(population)
            .await
            .map_err(|e| {
                error!("Evaluator failed at generation {gen}: {e}");
                e
            })
    }

    /// Records generation fitness metrics to stdout and logs, updating best candidate if improved,
    /// and dispatches structured telemetry to the sink if configured.
    async fn record_generation_stats(
        &self,
        gen: usize,
        elapsed_sec: f64,
        population: &[Vec<f64>],
        fitnesses: &[f64],
        best_ever: &mut f64,
        best_genome: &mut Vec<f64>,
        curr_snap: Option<crate::evaluator::tier_metrics::TierMetricsSnapshot>,
        prev_snap: Option<crate::evaluator::tier_metrics::TierMetricsSnapshot>,
    ) {
        let candidate = find_best_candidate(population, fitnesses);
        let (gen_best, avg, worst, std_dev) = compute_fitness_stats(fitnesses);

        if let Some((genes, fitness)) = candidate {
            if fitness > *best_ever || best_genome.is_empty() {
                *best_ever = fitness;
                *best_genome = genes.to_vec();
            }
        }

        info!(
            gen = gen,
            best = gen_best,
            avg = avg,
            best_ever = *best_ever,
            best_genome = ?best_genome,
            "Gen {gen}: best={gen_best:.4}, avg={avg:.4}, best_ever={best_ever:.4}, best_genome={best_genome:?}"
        );

        if let Some(sink) = self.telemetry {
            let gene_variance = compute_gene_variance(population, self.cfg.genes_len);
            let entropy = compute_population_entropy(population, self.cfg.genes_len, 10);
            let pop_snapshot = if self.cfg.record_population {
                Some(population.to_vec())
            } else {
                None
            };

            let (tier1_exact_hits, tier2_surrogate_hits, tier3_cfd_evals, tier_bypass_ratio) = match curr_snap {
                Some(curr) => {
                    let prev = prev_snap.unwrap_or(crate::evaluator::tier_metrics::TierMetricsSnapshot {
                        tier1_exact_hits: 0,
                        tier2_surrogate_hits: 0,
                        tier3_simulator_evals: 0,
                        total_evaluations: 0,
                    });
                    let t1 = curr.tier1_exact_hits.saturating_sub(prev.tier1_exact_hits) as u64;
                    let t2 = curr.tier2_surrogate_hits.saturating_sub(prev.tier2_surrogate_hits) as u64;
                    let t3 = curr.tier3_simulator_evals.saturating_sub(prev.tier3_simulator_evals) as u64;
                    let total = t1 + t2 + t3;
                    let bypass = if total > 0 { (t1 + t2) as f64 / total as f64 } else { 0.0 };
                    (Some(t1), Some(t2), Some(t3), Some(bypass))
                }
                None => (None, None, None, None),
            };

            let record = GenerationRecord {
                generation: gen,
                elapsed_sec,
                best_fitness: gen_best,
                avg_fitness: avg,
                worst_fitness: worst,
                std_fitness: std_dev,
                best_genome: best_genome.clone(),
                gene_variance,
                entropy,
                population: pop_snapshot,
                tier1_exact_hits,
                tier2_surrogate_hits,
                tier3_cfd_evals,
                tier_bypass_ratio,
            };
            sink.record(record).await;
        }
    }


    /// Emigrates individuals to ring successor if due.
    async fn handle_emigration(
        &self,
        gen: usize,
        population: &[Vec<f64>],
        fitnesses: &[f64],
    ) {
        if let Some(mig) = self.migration {
            if let Err(e) = mig.maybe_emigrate(gen, population, fitnesses).await {
                warn!("Migration (emigrate) failed at gen {gen}: {e}");
            }
        }
    }

    /// Selects survivors and reproduces the next generation.
    fn advance_population(
        &self,
        population: &[Vec<f64>],
        fitnesses: &[f64],
        rng: &mut StdRng,
        normal: &Normal<f64>,
    ) -> Vec<Vec<f64>> {
        let survivors = select_survivors(population, fitnesses, self.cfg);
        next_generation(&survivors, self.cfg, rng, normal)
    }
}
