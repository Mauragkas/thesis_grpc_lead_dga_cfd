# Genetic Algorithm Module

Core generational loop execution and pure genetic operators.

## Files and Code Structure

### 1. `algorithm.rs`
- **`GaRunner<'a>`**:
  - **Fields**:
    - `pub cfg: &'a GaConfig`: GA tunables (population size, mutation sigma, generations).
    - `pub evaluator: &'a dyn Evaluator`: Evaluation port.
    - `pub store: &'a dyn GeneStore`: Local exact cache and eviction port.
    - `pub neighbor_store: Option<&'a dyn NeighborStore>`: Optional LEAD DHT multi-probe index.
    - `pub migration: Option<&'a dyn MigrationHook>`: Optional island-model migration hook.
  - **Methods**:
    - `pub async fn run(&self, rng: &mut StdRng) -> Result<f64, tonic::Status>`:
      Executes the full generational lifecycle:
      1. Generates initial random population (`random_population`).
      2. In each generation:
         - Drains immigrants from `migration` and appends them to population.
         - Queries `store.lookup_exact` to find cached fitness values.
         - Filters uncached individuals and invokes `evaluator.evaluate_population`.
         - Stores newly evaluated individuals in `store` and optionally `neighbor_store`.
         - Evicts expired records via `store.evict_expired(gen)`.
         - Computes generation best, average, and updates `best_ever`.
         - Emigrates top individuals via `migration.maybe_emigrate`.
         - Selects survivors (`select_survivors`) and breeds next generation (`next_generation`).
      3. Returns the highest fitness achieved (`best_ever`).

---

### 2. `operators.rs`
Pure functional operators isolated from storage and gRPC layers:

- **`pub fn random_population<R: rand::Rng>(rng: &mut R, cfg: &GaConfig) -> Vec<Vec<f64>>`**:
  - Generates a `pop_size x genes_len` matrix of uniform random numbers in $[0.0, 1.0]$.
- **`pub fn select_survivors(population: &[Vec<f64>], fitnesses: &[f64], cfg: &GaConfig) -> Vec<Vec<f64>>`**:
  - $(\mu + \lambda)$ elitist survivor selection. Sorts individuals descending by fitness and returns the top $\max(2, \lfloor \text{pop\_size} \cdot \text{elite\_frac} \rfloor)$ individuals.
- **`pub fn next_generation<R: rand::Rng>(survivors: &[Vec<f64>], cfg: &GaConfig, rng: &mut R, normal: &rand_distr::Normal<f64>) -> Vec<Vec<f64>>`**:
  - Re-populates up to `pop_size` by selecting a random survivor parent, adding Gaussian noise $\mathcal{N}(0, \sigma^2)$ to each gene, and clipping to $[0.0, 1.0]$.
