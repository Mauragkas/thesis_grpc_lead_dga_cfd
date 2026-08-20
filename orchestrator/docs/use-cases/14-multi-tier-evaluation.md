# Use Case: Multi-Tier Evaluation ($\epsilon$-Bypass) Pipeline

## Description

Evaluates a population of candidate aircraft designs using a 3-tier hierarchical evaluation pipeline to minimize expensive aerodynamic simulations by exploiting spatial proximity in the design space.

```
                         Individual x*
                               │
                               ▼
                [Query 1st-NN & Distance d_min]
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        d_min < ε_exact   ε_exact ≤ d_min ≤ R   d_min > R
        [Tier 1: Hit]    [Tier 2: MLP Surrogate] [Tier 3: True Sim]
        Return y_NN      gRPC Call to           Dispatch to Worker
        (0 FLOPs)        surrogate-node         Insert (x*, y_true) into LEAD
                         (Active MLP Model)     & Ingest to Surrogate Window
```

## Primary Actor

- `MultiTierEvaluator`

## Supporting Systems

- `InMemoryGeneStore` (local exact match and fast k-NN cache)
- `HilbertNeighborStore` / `lead` DHT (distributed multi-probe spatial index)
- `GrpcSurrogateClient` / `surrogate-node` (fast MLP surrogate inference)
- `GrpcEvaluator` / `worker` simulator pool (high-fidelity AeroSandbox CFD evaluation)
- `TierMetricsTracker` (telemetry and bypass percentage computation)

## Main Steps

### Phase 1: Spatial Proximity & Tier Partitioning
1. For each individual $\mathbf{x}^*$ in the population:
   a. Check local in-memory store for exact bit-identical match. If found, mark as **Tier 1 Exact Hit** (fitness $y = y_{\text{stored}}$).
   b. Query the nearest neighbor $\mathbf{x}_{\text{NN}}$ from the local store or LEAD DHT.
   c. Compute Euclidean distance $d_{\min} = \|\mathbf{x}^* - \mathbf{x}_{\text{NN}}\|_2$.
   d. If $d_{\min} < \epsilon_{\text{exact}}$: Mark as **Tier 1 Cache Hit** ($y = y_{\text{NN}}$, 0 FLOPs).
   e. Else if $\epsilon_{\text{exact}} \le d_{\min} \le R$: Mark as **Tier 2 Candidate** (interpolation zone).
   f. Else ($d_{\min} > R$ or no known neighbors): Mark as **Tier 3 Candidate** (unexplored void region).

### Phase 2: Tier 2 Surrogate Inference & Fallback
2. If there are Tier 2 candidates:
   a. Dispatch batch to `SurrogateClient::predict_batch`.
   b. If the surrogate returns `Some(predictions)`: Assign predicted fitness values and record Tier 2 hits.
   c. If the surrogate returns `None` (cold-start / not ready) or an error occurs: Fall back to Tier 3 by appending candidates to the Tier 3 queue.

### Phase 3: Tier 3 True Worker Simulation & Feedback Loop
3. If there are Tier 3 candidates:
   a. Dispatch batch to `GrpcEvaluator::evaluate_population` (AeroSandbox solver via Envoy).
   b. Assign ground-truth fitnesses $y_{\text{true}}$ and record Tier 3 evaluations.
   c. Persist $(\mathbf{x}^*, y_{\text{true}})$ into `InMemoryGeneStore` and `HilbertNeighborStore` (LEAD DHT).
   d. Asynchronously ingest valid $(\mathbf{x}^*, y_{\text{true}})$ pairs into `SurrogateClient::ingest_samples` to update the surrogate sliding window.

### Phase 4: Telemetry & Return
4. Record metrics snapshot (Tier 1 hits, Tier 2 surrogate hits, Tier 3 simulator evals, bypass ratio).
5. Return complete fitness vector to the GA runner.

## Postconditions

- Every individual in the population is assigned a valid fitness value.
- The spatial index and surrogate buffer are expanded with new ground-truth samples.
- Generation metrics capture the percentage of simulations bypassed.

## Failure Cases

- Worker simulator unavailable -> Retries up to `max_attempts`, then returns error.
- Surrogate service unavailable -> Graceful fallback to Tier 3 simulation without failing GA run.
- LEAD DHT unreachable -> Continues with local in-memory store.
