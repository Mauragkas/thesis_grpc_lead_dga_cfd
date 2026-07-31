use crate::config::GaConfig;
use crate::evaluator::Evaluator;
use crate::ga::operators::{next_generation, random_population, select_survivors};
use log::info;
use rand::rngs::StdRng;
use rand_distr::Normal;
use std::time::Instant;
use tonic::Status;

/// SRP: orchestrates the generational loop. Depends on the `Evaluator`
/// abstraction (DIP), never on gRPC types directly.
pub struct GaRunner<'a> {
    pub cfg: &'a GaConfig,
    pub evaluator: &'a dyn Evaluator,
}

impl<'a> GaRunner<'a> {
    pub async fn run(&self, rng: &mut StdRng) -> Result<f64, Status> {
        let start = Instant::now();
        let mut population = random_population(rng, self.cfg);
        let mut best_ever = f64::NEG_INFINITY;
        let normal = Normal::new(0.0, self.cfg.mut_sigma).unwrap();

        for gen in 1..=self.cfg.generations {
            info!("Evaluating generation {gen}/{}...", self.cfg.generations);
            let fitnesses = self.evaluator.evaluate_population(&population).await?;

            let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
            best_ever = best_ever.max(best);
            println!(
                "Gen {}/{} | Best: {:.4} | Avg: {:.4} | BestEver: {:.4}",
                gen, self.cfg.generations, best, avg, best_ever
            );

            let survivors = select_survivors(&population, &fitnesses, self.cfg);
            population = next_generation(&survivors, self.cfg, rng, &normal);
        }

        let elapsed = start.elapsed().as_secs_f64();
        println!("Finished GA run in {elapsed:.2}s. Best fitness: {best_ever:.4}");
        Ok(best_ever)
    }
}
