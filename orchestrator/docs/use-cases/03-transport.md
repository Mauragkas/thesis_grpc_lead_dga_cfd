# Use Case: Build Transport Endpoint and Wait for Readiness

## Description

Builds tonic transport endpoints from target URIs, polls for channel readiness before making gRPC calls, and provides a multi-layer transport resilience subsystem comprising a thread-safe Circuit Breaker and an exponential backoff Retry Policy with jitter.

## Primary actor

- Orchestrator (`GrpcEvaluator`, `GrpcLeadStore`, `GrpcSurrogateClient`, `GrpcRingClient`)

## Supporting systems

- `tonic::transport::{Endpoint, Channel}`
- `CircuitBreaker` (finite state machine: `Closed`, `Open`, `HalfOpen`)
- `RetryPolicy` (exponential backoff with jitter and retryability detection)
- Remote microservices (worker evaluator pool, LEAD DHT, surrogate node, ring peers)

## Main steps: Build Endpoint and Wait for Channel Readiness

1. Receive target URI string.
2. If it lacks `http://` or `https://`, prefix with `http://`.
3. Apply transport tunables from `TransportConfig`:
   - keep-alive while idle (`true`)
   - keep-alive timeout (`keep_alive_timeout`)
   - TCP keepalive (`tcp_keepalive`)
   - connect timeout (`connect_timeout`)
   - request timeout (`request_timeout`)
4. In `wait_for_channel(endpoint, deadline)`:
   - Attempt connection every 2s until `deadline` (derived from `channel_ready_deadline`).
   - If connected, return `Channel`; if deadline expires, return `Status::unavailable`.

## Main steps: Circuit Breaker State Machine

1. **`Closed` (Normal Operation)**:
   - Requests are permitted to execute immediately.
   - Successful RPCs invoke `record_success()`, resetting consecutive failure counts.
   - On transient failure, `record_failure()` is invoked. If failures reach `circuit_breaker_failure_threshold` (default 5), transition to `Open` and record timestamp.
2. **`Open` (Upstream Degraded / Scaling)**:
   - Requests are blocked from hammering the upstream.
   - Incoming requests wait up to `rpc_timeout` via `wait_until_ready()`.
   - Once `recovery_timeout` (default 1500ms) has elapsed, the circuit transitions to `HalfOpen`.
3. **`HalfOpen` (Probing Recovery)**:
   - Allows up to `half_open_probes` (default 2) trial requests through.
   - If probes succeed, transition back to `Closed`.
   - If a probe fails, immediately revert to `Open` with a new recovery cooldown.

## Main steps: Per-Individual Retry Policy

1. When evaluating individuals, requests are submitted to `eval_individual`.
2. For attempt $k \in [1..\text{max\_attempts}]$:
   - Await circuit breaker readiness (`wait_until_ready`).
   - Dispatch RPC with `rpc_timeout`.
   - If success: return fitness and record circuit breaker success.
   - If transient error (e.g. `Unavailable`, `DeadlineExceeded`):
     - Record circuit breaker failure.
     - Calculate exponential backoff: $\text{backoff} = \min(\text{max\_backoff}, \text{initial\_backoff} \times 2^{k-1}) \times (1 \pm \text{jitter})$.
     - Sleep and retry.
3. If all attempts are exhausted:
   - If `fallback_penalty_on_exhaustion` is enabled: log warning and assign penalty fitness `-1e9`.
   - Otherwise: return final error `Status`.

## Postconditions

- Live `Channel` connections are guarded against thundering herds and cascading worker failures during dynamic scaling.
- Transients are absorbed smoothly by backoff retries without failing entire GA generations.

## Failure cases

- Invalid URI format or connection deadline exceeded during bootstrap.
- Upstream permanently dead: Circuit breaker remains open and exhausted retries return error or fallback penalty.
