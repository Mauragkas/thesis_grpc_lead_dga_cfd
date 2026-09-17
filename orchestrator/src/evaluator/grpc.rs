use crate::config::TransportConfig;
use crate::evaluator::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use crate::evaluator::r#trait::Evaluator;
use crate::evaluator::retry::RetryPolicy;
use crate::proto::eval::{evaluator_client::EvaluatorClient, BatchRequest, Individual};
use std::sync::Arc;
use tonic::{transport::Channel, Request, Status};
use tracing::{debug, error, info, warn};

/// Concrete gRPC-backed evaluator with per-individual retry policies and circuit breaking.
/// Low-level detail owned by this module; high-level GA code never names `EvaluatorClient` directly (DIP).
pub struct GrpcEvaluator {
    client: EvaluatorClient<Channel>,
    cfg: TransportConfig,
    batch_size: usize,
    circuit_breaker: Arc<CircuitBreaker>,
    retry_policy: RetryPolicy,
}

impl GrpcEvaluator {
    pub fn new(client: EvaluatorClient<Channel>, cfg: TransportConfig, batch_size: usize) -> Self {
        let cb_config = CircuitBreakerConfig {
            failure_threshold: cfg.circuit_breaker_failure_threshold,
            recovery_timeout: cfg.circuit_breaker_recovery_timeout,
            half_open_probes: cfg.circuit_breaker_half_open_probes,
        };
        let retry_policy = RetryPolicy::new(
            cfg.max_individual_retries,
            cfg.initial_retry_backoff,
            cfg.max_retry_backoff,
            cfg.retry_jitter_factor,
        );
        Self {
            client,
            cfg,
            batch_size,
            circuit_breaker: Arc::new(CircuitBreaker::new(cb_config)),
            retry_policy,
        }
    }

    /// Construct with explicit circuit breaker and retry policy (useful for testing and dependency injection).
    pub fn with_resilience(
        client: EvaluatorClient<Channel>,
        cfg: TransportConfig,
        batch_size: usize,
        circuit_breaker: Arc<CircuitBreaker>,
        retry_policy: RetryPolicy,
    ) -> Self {
        Self {
            client,
            cfg,
            batch_size,
            circuit_breaker,
            retry_policy,
        }
    }

    pub fn circuit_breaker(&self) -> &Arc<CircuitBreaker> {
        &self.circuit_breaker
    }

    pub fn retry_policy(&self) -> &RetryPolicy {
        &self.retry_policy
    }

    /// Evaluates a single individual with per-individual retries and circuit breaker protection.
    pub async fn eval_individual(
        client: &mut EvaluatorClient<Channel>,
        circuit_breaker: &Arc<CircuitBreaker>,
        retry_policy: &RetryPolicy,
        cfg: &TransportConfig,
        genes: &[f64],
    ) -> Result<f64, Status> {
        let mut last_err: Option<Status> = None;

        for attempt in 1..=retry_policy.max_attempts {
            // Guard with circuit breaker: wait for cooling period if circuit is Open
            circuit_breaker.wait_until_ready(cfg.rpc_timeout).await?;

            let req = Individual {
                genes: genes.to_vec(),
            };
            let call = client.evaluate(Request::new(req));

            match tokio::time::timeout(cfg.rpc_timeout, call).await {
                Ok(Ok(resp)) => {
                    circuit_breaker.record_success();
                    debug!("Individual evaluated on attempt {attempt}");
                    return Ok(resp.into_inner().fitness);
                }
                Ok(Err(e)) if RetryPolicy::is_retryable(&e) => {
                    circuit_breaker.record_failure();
                    warn!(
                        "Individual eval transient error (attempt {}/{}): {e}",
                        attempt, retry_policy.max_attempts
                    );
                    last_err = Some(e);
                    if attempt < retry_policy.max_attempts {
                        let backoff = retry_policy.backoff_for_attempt(attempt);
                        tokio::time::sleep(backoff).await;
                    }
                }
                Ok(Err(e)) => {
                    error!("Non-retryable gRPC error during individual eval: {e}");
                    return Err(e);
                }
                Err(_) => {
                    circuit_breaker.record_failure();
                    warn!(
                        "Individual eval timed out after {:?} (attempt {}/{})",
                        cfg.rpc_timeout, attempt, retry_policy.max_attempts
                    );
                    last_err = Some(Status::unavailable("rpc timed out"));
                    if attempt < retry_policy.max_attempts {
                        let backoff = retry_policy.backoff_for_attempt(attempt);
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }

        warn!(
            "Exhausted all {} individual retry attempts; last error: {:?}",
            retry_policy.max_attempts, last_err
        );

        if cfg.fallback_penalty_on_exhaustion {
            warn!("Assigning penalty fitness -1e9 due to exhausted retries");
            Ok(-1e9)
        } else {
            Err(last_err.unwrap_or_else(|| Status::unavailable("exhausted retries")))
        }
    }

    /// Evaluates a chunk of individuals. If batch execution fails with a transient error,
    /// falls back to per-individual evaluation with circuit breaker protection.
    async fn eval_chunk(
        mut client: EvaluatorClient<Channel>,
        circuit_breaker: Arc<CircuitBreaker>,
        retry_policy: RetryPolicy,
        cfg: TransportConfig,
        chunk: Vec<Vec<f64>>,
    ) -> Result<Vec<f64>, Status> {
        if chunk.len() == 1 {
            let fit = Self::eval_individual(
                &mut client,
                &circuit_breaker,
                &retry_policy,
                &cfg,
                &chunk[0],
            )
            .await?;
            return Ok(vec![fit]);
        }

        // Try batch evaluation first
        if circuit_breaker.wait_until_ready(cfg.rpc_timeout).await.is_ok() {
            let req = BatchRequest {
                individuals: chunk
                    .iter()
                    .map(|g| Individual { genes: g.clone() })
                    .collect(),
            };

            let call = client.evaluate_batch(Request::new(req));
            match tokio::time::timeout(cfg.rpc_timeout, call).await {
                Ok(Ok(resp)) => {
                    circuit_breaker.record_success();
                    return Ok(resp
                        .into_inner()
                        .results
                        .into_iter()
                        .map(|r| r.fitness)
                        .collect());
                }
                Ok(Err(e)) if RetryPolicy::is_retryable(&e) => {
                    circuit_breaker.record_failure();
                    warn!(
                        "Batch evaluation failed with transient error: {e}; falling back to per-individual evaluation",
                    );
                }
                Ok(Err(e)) => {
                    error!("Non-retryable gRPC error during batch eval: {e}");
                    return Err(e);
                }
                Err(_) => {
                    circuit_breaker.record_failure();
                    warn!(
                        "Batch evaluation timed out after {:?}; falling back to per-individual evaluation",
                        cfg.rpc_timeout
                    );
                }
            }
        }

        // Fallback: evaluate individuals in this chunk independently
        debug!(
            "Evaluating {} individuals individually following batch degradation",
            chunk.len()
        );
        let mut results = Vec::with_capacity(chunk.len());
        for ind in chunk {
            let fit = Self::eval_individual(
                &mut client,
                &circuit_breaker,
                &retry_policy,
                &cfg,
                &ind,
            )
            .await?;
            results.push(fit);
        }
        Ok(results)
    }
}

#[async_trait::async_trait]
impl Evaluator for GrpcEvaluator {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
        let chunks: Vec<&[Vec<f64>]> = population.chunks(self.batch_size).collect();
        info!(
            "Evaluating {} individuals in {} batch(es) (batch_size={})",
            population.len(),
            chunks.len(),
            self.batch_size
        );
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let c = self.client.clone();
            let owned: Vec<Vec<f64>> = chunk.to_vec();
            let cfg = self.cfg.clone();
            let cb = self.circuit_breaker.clone();
            let rp = self.retry_policy.clone();
            handles.push(tokio::spawn(async move {
                Self::eval_chunk(c, cb, rp, cfg, owned).await
            }));
        }

        let mut flat = Vec::with_capacity(population.len());
        for h in handles {
            flat.extend(h.await.map_err(|e| {
                error!("Batch task join error: {e}");
                Status::internal(format!("join error: {e}"))
            })??);
        }
        Ok(flat)
    }
}
