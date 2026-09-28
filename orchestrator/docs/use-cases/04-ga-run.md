# Use Case: Run Genetic Algorithm
 
## Description
 
Coordinates the full GA generational lifecycle: initial population generation, immigrant integration, local cache lookup, batch evaluation over gRPC, multi-probe LEAD DHT neighbor storage, cache eviction, metric tracking, island emigration, survivor selection, and breeding.
 
## Primary actor
 
- `GaRunner`
 
## Supporting systems
 
- `Evaluator` (`GrpcEvaluator`)
- `GeneStore` (`InMemoryGeneStore`)
- Optional `NeighborStore` (`HilbertNeighborStore`)
- Optional `MigrationHook` (`LeadMigration`)
- GA operators (`random_population`, `select_survivors`, `next_generation`)
 
## Preconditions
 
- `GaConfig` is loaded.
- Seeded `StdRng` is initialized.
- Evaluator and local gene store are available.
- Optional neighbor store and migration hooks are configured.
 
## Main steps
 
1. Generate the initial random population within normalized `[0, 1]` bounds.
2. Initialize `Normal` distribution for mutations.
3. Initialize `ProgressTracker::new(stagnation_patience, min_improvement, min_generations)`.
4. For each generation `gen` from `1` to `max_generations`:
   1. **Immigration**: If `MigrationHook` is configured, drain immigrants from the buffer and integrate them into the population.
   2. **Cache check**: Query `store.lookup_exact(genes, gen)` for each individual.
   3. **Filter uncached**: Collect all cache misses that require remote evaluation.
   4. **Evaluate**: Call `evaluator.evaluate_population(&uncached)` for newly encountered genes.
   5. **Store locally**: Insert newly evaluated individuals and their fitness into the local `GeneStore`.
   6. **Store in LEAD DHT**: If `NeighborStore` is configured, persist each newly evaluated individual under multi-probe Hilbert keys.
   7. **Eviction**: Call `store.evict_expired(gen)` to drop records exceeding the TTL window.
   8. **Statistics**: Compute generation best and average fitness, update all-time `best_ever` and `best_genome`, and log metrics.
   9. **Emigration**: If `MigrationHook` is configured and `gen` matches the migration interval, select top individuals and send them to the ring successor.
   10. **Convergence Check**: Update `tracker.update(best_ever, gen)`. If progress has stagnated ($< \text{min\_improvement}$) for $\ge \text{stagnation\_patience}$ consecutive generations and $\text{gen} \ge \text{min\_generations}$, log early convergence and break out of the generational loop.
   11. **Selection**: Call `select_survivors()` to keep the top `elite_frac` portion.
   12. **Breeding**: Call `next_generation()` to mutate survivors into a full new population.
5. Return the `GaResult` containing `best_fitness` and `best_genome` achieved across the run.
 
## Postconditions
 
- The GA run completes (either reaching `max_generations` or terminating early upon stagnation convergence).
- The `GaResult` with the best fitness score and genome parameter vector is returned to `main`.
 
## Failure cases
 
- Evaluator error during batch evaluation.
- Unrecoverable transport timeout or server failure.
- Migration failures are logged as warnings and do not abort the GA loop.
- LEAD DHT persistence failures are logged as warnings and do not abort the GA loop.

