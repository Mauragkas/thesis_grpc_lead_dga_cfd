# Use Case: Evaluate a Population Through gRPC

## Description

Sends one or more individuals to the evaluator service over gRPC and returns a fitness value for each individual.

## Primary actor

- `GrpcEvaluator`

## Supporting systems

- `eval.Evaluator` gRPC service

## Main steps

1. Split the population into chunks using `batch_size`.
2. Clone the gRPC client per chunk.
3. Build a `BatchRequest` for each chunk.
4. Call `EvaluateBatch`.
5. If the call succeeds, extract fitness values from the response.
6. If the call returns `Unavailable`, retry after a delay.
7. If the call times out, retry after a delay.
8. Stop after `max_attempts`.
9. Flatten all chunk results into a single ordered fitness vector.

## Postconditions

- One fitness value exists for each individual in the input population.

## Failure cases

- Deadline exceeded
- Upstream unavailable
- Non-retryable gRPC error
- Join error from spawned task
