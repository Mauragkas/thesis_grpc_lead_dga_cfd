use crate::config::{clip, GaConfig};
use rand_distr::{Distribution, Normal};

/// SRP: initial random population in normalized gene space.
pub fn random_population<R: rand::Rng>(rng: &mut R, cfg: &GaConfig) -> Vec<Vec<f64>> {
    (0..cfg.pop_size)
        .map(|_| (0..cfg.genes_len).map(|_| rng.gen::<f64>()).collect())
        .collect()
}

/// SRP: (μ + λ)-style survivor selection — top `elite_frac` by fitness.
pub fn select_survivors(
    population: &[Vec<f64>],
    fitnesses: &[f64],
    cfg: &GaConfig,
) -> Vec<Vec<f64>> {
    let n_survivors = std::cmp::max(2, (cfg.pop_size as f64 * cfg.elite_frac) as usize);
    let mut indexed: Vec<usize> = (0..population.len()).collect();
    indexed.sort_by(|&a, &b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap().reverse());
    indexed
        .iter()
        .take(n_survivors)
        .map(|&i| population[i].clone())
        .collect()
}

/// SRP: produces next generation by mutating a random survivor.
/// OCP: a `Breeder` trait could generalise this, but per the guardrails we
/// keep it concrete until a second strategy is actually needed.
pub fn next_generation<R: rand::Rng>(
    survivors: &[Vec<f64>],
    cfg: &GaConfig,
    rng: &mut R,
    normal: &Normal<f64>,
) -> Vec<Vec<f64>> {
    let mut next = survivors.to_vec();
    while next.len() < cfg.pop_size {
        let parent = &survivors[rng.gen_range(0..survivors.len())];
        let child: Vec<f64> = parent
            .iter()
            .map(|&g| clip(g + normal.sample(rng)))
            .collect();
        next.push(child);
    }
    next
}
