# Use Case: Use Exact Gene Cache

## Description

Avoids re-evaluating individuals that have already been evaluated by checking the exact gene cache before calling the evaluator.

## Primary actor

- `GaRunner`

## Supporting systems

- `GeneStore`

## Main steps

1. For each individual in the current population:
   1. Call `lookup_exact`.
   2. If a fitness exists, reuse it.
   3. If not, mark the individual as uncached.
2. Send only uncached individuals to the evaluator.
3. After evaluation, store the new result in the gene store.

## Postconditions

- Duplicate individuals are not re-evaluated.
- Evaluator load is reduced.

## Notes

- Exact match means bit-identical `Vec<f64>` values in the current implementation.
- Access refreshes the record’s `last_accessed_gen`.
