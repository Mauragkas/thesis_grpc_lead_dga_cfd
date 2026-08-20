use crate::config::GaConfig;
use crate::evaluator::Evaluator;
use crate::ga::operators::{next_generation, random_population, select_survivors};
use crate::gene_store::GeneStore;
use crate::migration::MigrationHook;
use crate::neighbor_store::NeighborStore;
use rand::rngs::StdRng;
use rand_distr::Normal;
use std::time::Instant;
use tonic::Status;
use tracing::{error, info, warn};

/// SRP: orchestrates the generational loop. Depends on the `Evaluator`,
/// `GeneStore`, `NeighborStore`, and `MigrationHook` abstractions (DIP), never on concrete
/// impls directly.
pub struct GaRunner<'a> {
    pub cfg: &'a GaConfig,
    pub evaluator: &'a dyn Evaluator,
    pub store: &'a dyn GeneStore,
    pub neighbor_store: Option<&'a dyn NeighborStore>,
    pub migration: Option<&'a dyn MigrationHook>,
}

impl<'a> GaRunner<'a> {
    pub async fn run(&self, rng: &mut StdRng) -> Result<f64, Status> {
        let start = Instant::now();
        info!(
            "GA run starting: pop_size={}, genes_len={}, generations={}, mut_sigma={}, elite_frac={}, batch_size={}",
            self.cfg.pop_size,
            self.cfg.genes_len,
            self.cfg.generations,
            self.cfg.mut_sigma,
            self.cfg.elite_frac,
            self.cfg.batch_size
        );
        let mut population = random_population(rng, self.cfg);
        let mut best_ever = f64::NEG_INFINITY;
        let normal = Normal::new(0.0, self.cfg.mut_sigma).unwrap();

        for gen in 1..=self.cfg.generations {
            info!("Evaluating generation {gen}/{}...", self.cfg.generations);

            self.integrate_immigrants(&mut population, gen).await;

            let fitnesses = self.evaluate_generation(&population, gen).await?;

            self.store.evict_expired(gen).await;

            self.record_generation_stats(gen, &fitnesses, &mut best_ever);

            self.handle_emigration(gen, &population, &fitnesses).await;

            population = self.advance_population(&population, &fitnesses, rng, &normal);
        }

        let elapsed = start.elapsed().as_secs_f64();
        info!("GA run finished in {elapsed:.2}s; best fitness: {best_ever:.4}");
        println!("Finished GA run in {elapsed:.2}s. Best fitness: {best_ever:.4}");
        Ok(best_ever)
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

    /// Records generation fitness metrics to stdout and logs.
    fn record_generation_stats(
        &self,
        gen: usize,
        fitnesses: &[f64],
        best_ever: &mut f64,
    ) {
        let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        *best_ever = (*best_ever).max(best);
        println!(
            "Gen {}/{} | Best: {:.4} | Avg: {:.4} | BestEver: {:.4}",
            gen, self.cfg.generations, best, avg, *best_ever
        );
        info!(
            "Gen {gen}: best={best:.4}, avg={avg:.4}, best_ever={:.4}",
            *best_ever
        );
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
