# Use Case: GA Operators

## Description

Provides the pure functions that build and transform populations during the genetic algorithm.

## Included operations

- `random_population`
- `select_survivors`
- `next_generation`

## Use case: Generate random population

### Steps
1. Read `pop_size` and `genes_len`.
2. Create `pop_size` individuals.
3. Fill each individual with random genes in `[0, 1]`.

### Result
- Initial population of normalized genes.

## Use case: Select survivors

### Steps
1. Pair each individual with its fitness.
2. Sort from highest to lowest fitness.
3. Select the top `elite_frac` portion.
4. Ensure at least two survivors are kept.

### Result
- Survivor pool for breeding.

## Use case: Breed next generation

### Steps
1. Start with the survivors.
2. Randomly choose parents from the survivor pool.
3. Add Gaussian mutation to each gene.
4. Clip each gene into `[0, 1]`.
5. Repeat until population size is restored.

### Result
- Next generation population.

## Notes

- These functions are pure except for their RNG input.
- They are testable independently from the gRPC and storage layers.
