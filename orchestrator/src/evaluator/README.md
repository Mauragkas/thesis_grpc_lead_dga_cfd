# Evaluator Module

Defines the evaluation contract, the **Multi-Tier ($\epsilon$-Bypass) Evaluation Pipeline**, tier metrics tracking, and the gRPC batch worker simulator client.

## Multi-Tier Evaluation ($\epsilon$-Bypass)

The `MultiTierEvaluator` implements an active, hierarchical evaluation strategy to maximize throughput and minimize expensive aerodynamic solver calls:

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

## Files and Code Structure

### 1. `trait.rs`
- **`Evaluator`** (async trait):
  ```rust
  #[async_trait::async_trait]
  pub trait Evaluator: Send + Sync {
      async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, tonic::Status>;
  }
  ```
  Primary port (DIP) used by `GaRunner` to evaluate candidate populations without depending on concrete transports or storage.

---

### 2. `multi_tier.rs`
- **`MultiTierEvaluator`**:
  - Implements `Evaluator`.
  - **Dependencies**: Injects `Arc<dyn Evaluator>` (simulator), `Arc<dyn GeneStore>` (exact cache), `Option<Arc<dyn NeighborStore>>` (LEAD DHT), `Option<Arc<dyn SurrogateClient>>` (surrogate node), `TierConfig`, and `TierMetricsTracker`.
  - **Routing Flow**:
    - **Tier 1 ($d_{\min} < \epsilon_{\text{exact}}$)**: Returns closest stored neighbor fitness directly (**0 FLOPs**).
    - **Tier 2 ($\epsilon_{\text{exact}} \le d_{\min} \le R$)**: Calls `surrogate.predict_batch` using the active MLP surrogate model. If surrogate is not ready (cold start), falls back to Tier 3.
    - **Tier 3 ($d_{\min} > R$ or fallback)**: Evaluates via `worker` simulator, persists $(\mathbf{x}^*, y_{\text{true}})$ into local and LEAD stores, and feeds samples to `surrogate_node`'s sliding window.

---

### 3. `tier_metrics.rs`
- **`TierMetricsTracker`**: Thread-safe atomic tracker recording:
  - `tier1_exact_hits`: Count of 0-FLOP cache hits.
  - `tier2_surrogate_hits`: Count of fast MLP surrogate predictions.
  - `tier3_simulator_evals`: Count of true worker simulator calls.
  - `total_evaluations`: Total individuals evaluated.
- **`TierMetricsSnapshot`**: Computed percentages (`tier1_ratio`, `tier2_ratio`, `tier3_ratio`, `bypass_ratio`).

---

### 4. `grpc.rs`
- **`GrpcEvaluator`**:
  - Concrete gRPC client communicating with the remote AeroSandbox worker pool behind Envoy load balancer.
  - Handles batching, request timeouts, and exponential retry backoff.
