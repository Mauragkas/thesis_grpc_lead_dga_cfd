use crate::config::GaConfig;
use crate::evaluator::Evaluator;
use crate::ga::operators::{next_generation, random_population, select_survivors};
use crate::gene_store::GeneStore;
use crate::lead_store::{gene_key, GenePayload, LeadStore};
use log::{info, warn};
use rand::rngs::StdRng;
use rand_distr::Normal;
use std::time::Instant;
use tonic::Status;

/// SRP: orchestrates the generational loop. Depends on the `Evaluator`
/// and `GeneStore` abstractions (DIP), never on concrete impls directly.
pub struct GaRunner<'a> {
    pub cfg: &'a GaConfig,
    pub evaluator: &'a dyn Evaluator,
    pub store: &'a dyn GeneStore,
    pub lead_store: Option<&'a dyn LeadStore>,
}

impl<'a> GaRunner<'a> {
    pub async fn run(&self, rng: &mut StdRng) -> Result<f64, Status> {
        let start = Instant::now();
        let mut population = random_population(rng, self.cfg);
        let mut best_ever = f64::NEG_INFINITY;
        let normal = Normal::new(0.0, self.cfg.mut_sigma).unwrap();

        for gen in 1..=self.cfg.generations {
            info!("Evaluating generation {gen}/{}...", self.cfg.generations);

            let mut fitnesses = vec![f64::NEG_INFINITY; population.len()];
            let mut uncached_idx: Vec<usize> = Vec::new();
            let mut uncached: Vec<Vec<f64>> = Vec::new();

            for (i, genes) in population.iter().enumerate() {
                match self.store.lookup_exact(genes, gen).await {
                    Some(fit) => fitnesses[i] = fit,
                    None => {
                        uncached_idx.push(i);
                        uncached.push(genes.clone());
                    }
                }
            }

            let hits = population.len() - uncached.len();
            if hits > 0 {
                info!(
                    "Gen {gen}: {hits} exact cache hits, {} sent to evaluator",
                    uncached.len()
                );
            }

            // Only RPC the individuals we genuinely need to evaluate.
            if !uncached.is_empty() {
                let fresh = self.evaluator.evaluate_population(&uncached).await?;
                for ((&i, g), f) in uncached_idx.iter().zip(uncached.iter()).zip(fresh.iter()) {
                    fitnesses[i] = *f;
                    self.store.store(g.clone(), *f, gen).await;

                    if let Some(ls) = self.lead_store {
                        let key = gene_key(g);
                        let payload = GenePayload {
                            genes: g.clone(),
                            fitness: *f,
                            generation: gen,
                        };
                        let value = serde_json::to_string(&payload).unwrap_or_default();
                        if let Err(e) = ls.store_gene(&key, &value).await {
                            warn!("lead store failed (gen {gen}): {e}");
                        }
                    }
                }
            }

            // Drop records not retrieved within the TTL window.
            self.store.evict_expired(gen).await;

            let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
            best_ever = best_ever.max(best);
            println!(
                "Gen {}/{} | Best: {:.4} | Avg: {:.4} | BestEver: {:.4} | Cache hits: {}",
                gen, self.cfg.generations, best, avg, best_ever, hits
            );

            let survivors = select_survivors(&population, &fitnesses, self.cfg);
            population = next_generation(&survivors, self.cfg, rng, &normal);
        }

        let elapsed = start.elapsed().as_secs_f64();
        println!("Finished GA run in {elapsed:.2}s. Best fitness: {best_ever:.4}");
        Ok(best_ever)
    }
}
