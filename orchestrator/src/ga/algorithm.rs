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

            // --- Drain immigrants and integrate before evaluation ---
            if let Some(mig) = self.migration {
                let immigrants = mig.drain_immigrants().await;
                if !immigrants.is_empty() {
                    info!(
                        "Gen {gen}: integrating {} immigrants into population",
                        immigrants.len()
                    );
                    // Replace the worst individuals with immigrants.
                    let indexed: Vec<usize> = (0..population.len()).collect();
                    // Sort ascending by current best-known fitness (we don't
                    // have fitnesses yet this gen, so just replace from the
                    // back — the last slots are arbitrary in a fresh pop).
                    // A smarter approach: evaluate first, then replace worst.
                    // For simplicity, append immigrants and trim to pop_size.
                    for im in immigrants {
                        population.push(im.genes);
                    }
                    population.truncate(self.cfg.pop_size);
                    let _ = indexed;
                }
            }

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
                let fresh = self
                    .evaluator
                    .evaluate_population(&uncached)
                    .await
                    .map_err(|e| {
                        error!("Evaluator failed at generation {gen}: {e}");
                        e
                    })?;
                for ((&i, g), f) in uncached_idx.iter().zip(uncached.iter()).zip(fresh.iter()) {
                    fitnesses[i] = *f;
                    self.store.store(g.clone(), *f, gen).await;

                    if let Some(ns) = self.neighbor_store {
                        if let Err(e) = ns.store(g, *f, gen).await {
                            warn!("neighbor store failed (gen {gen}): {e}");
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
            info!(
                "Gen {gen}: best={best:.4}, avg={avg:.4}, best_ever={best_ever:.4}, cache_hits={hits}"
            );

            // --- Emigrate after evaluation ---
            if let Some(mig) = self.migration {
                if let Err(e) = mig.maybe_emigrate(gen, &population, &fitnesses).await {
                    warn!("Migration (emigrate) failed at gen {gen}: {e}");
                }
            }

            let survivors = select_survivors(&population, &fitnesses, self.cfg);
            population = next_generation(&survivors, self.cfg, rng, &normal);
        }

        let elapsed = start.elapsed().as_secs_f64();
        info!("GA run finished in {elapsed:.2}s; best fitness: {best_ever:.4}");
        println!("Finished GA run in {elapsed:.2}s. Best fitness: {best_ever:.4}");
        Ok(best_ever)
    }
}
