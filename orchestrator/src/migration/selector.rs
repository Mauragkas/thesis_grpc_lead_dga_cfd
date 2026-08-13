//! SRP / OCP: decides *which* individuals migrate. New strategies
//! (random, diversity-preserving, fitness-proportionate) implement
//! `MigrantSelector` without touching the migration orchestration.

use crate::migration::MigrantIndividual;

pub trait MigrantSelector: Send + Sync {
    fn select(
        &self,
        population: &[Vec<f64>],
        fitnesses: &[f64],
        count: usize,
    ) -> Vec<MigrantIndividual>;
}

/// Selects the top-K individuals by fitness. Simple and effective for
/// elitist migration.
#[derive(Debug, Clone, Copy, Default)]
pub struct TopKSelector;

impl MigrantSelector for TopKSelector {
    fn select(
        &self,
        population: &[Vec<f64>],
        fitnesses: &[f64],
        count: usize,
    ) -> Vec<MigrantIndividual> {
        let mut indexed: Vec<usize> = (0..population.len()).collect();
        indexed.sort_by(|&a, &b| {
            fitnesses[a]
                .partial_cmp(&fitnesses[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });
        indexed
            .iter()
            .take(count)
            .map(|&i| MigrantIndividual {
                genes: population[i].clone(),
                fitness: fitnesses[i],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_selects_highest_fitness() {
        let pop = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let fit = vec![10.0, 30.0, 20.0, 40.0];
        let migrants = TopKSelector.select(&pop, &fit, 2);
        assert_eq!(migrants.len(), 2);
        assert_eq!(migrants[0].genes, vec![3.0]); // fitness 40
        assert_eq!(migrants[1].genes, vec![1.0]); // fitness 30
    }
}
