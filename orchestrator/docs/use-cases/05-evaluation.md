# Use Case: Evaluate a Population Through gRPC

## Description

Evaluates a population of candidate designs against the worker simulation pool over gRPC using concurrent chunking, circuit-breaker guarded batch evaluation, and automatic degradation to per-individual retries with exponential backoff and jitter upon upstream transient failure.

## Primary actor

- `GrpcEvaluator`

## Supporting systems

- `eval.Evaluator` gRPC service (Envoy load balancer -> worker pool)
- `CircuitBreaker` (concurrency guard and fast degradation preventer)
- `RetryPolicy` (exponential backoff with jitter)

## Main steps

1. Split the population into chunks using `batch_size`.
2. For each chunk, spawn an asynchronous worker task via `tokio::spawn`:
   - **Single-Individual Optimization**: If `chunk.len() == 1`, directly execute `eval_individual`.
   - **Batch Execution**: If `chunk.len() > 1`:
     1. Await circuit breaker readiness: `circuit_breaker.wait_until_ready(rpc_timeout)`.
     2. Construct `BatchRequest` and invoke `EvaluateBatch` over gRPC.
     3. If the batch call succeeds within `rpc_timeout`:
        - Record success: `circuit_breaker.record_success()`.
        - Return extracted fitness vector.
     4. If the batch call fails with a retryable transient error (e.g. `Unavailable`) or times out:
        - Record failure: `circuit_breaker.record_failure()`.
        - Log warning: "Batch evaluation failed; falling back to per-individual evaluation".
        - Degrade execution: evaluate each individual in the chunk sequentially via `eval_individual`.
     5. If a non-retryable error occurs: return `Err(Status)`.
3. **Per-Individual Resilient Evaluation (`eval_individual`)**:
   - Loop for attempt $k \in [1..\text{max\_attempts}]$:
     1. Guard with circuit breaker: `circuit_breaker.wait_until_ready(rpc_timeout)`.
     2. Invoke `Evaluate` RPC for the single individual.
     3. If success: record circuit breaker success and return fitness.
     4. If retryable error:
        - Record circuit breaker failure.
        - Calculate backoff duration using `RetryPolicy::backoff_for_attempt(k)`.
        - Sleep and retry.
   - If all retry attempts are exhausted:
     - If `cfg.fallback_penalty_on_exhaustion` is enabled: log warning and return `-1e9` penalty fitness.
     - Otherwise: return final error `Status`.
4. Await all spawned chunk tasks and flatten results into the original ordered population vector.

## Postconditions

- Every candidate individual in the population is assigned a fitness value.
- Temporary upstream worker dropping or Envoy DNS propagation delays are absorbed gracefully by falling back to resilient per-individual retries.

## Failure cases

- Upstream permanently unreachable and retries exhausted without fallback penalty enabled.
- Non-retryable gRPC error returned by worker (e.g. `InvalidArgument`).
- Tokio task join error.
