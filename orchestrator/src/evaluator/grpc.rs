use crate::config::TransportConfig;
use crate::evaluator::r#trait::Evaluator;
use crate::proto::eval::{evaluator_client::EvaluatorClient, BatchRequest, Individual};
use log::info;
use tonic::{transport::Channel, Code, Request, Status};

/// Concrete gRPC-backed evaluator. Low-level detail owned by this module;
/// high-level GA code never names `EvaluatorClient` directly (DIP).
pub struct GrpcEvaluator {
    client: EvaluatorClient<Channel>,
    cfg: TransportConfig,
    batch_size: usize,
}

impl GrpcEvaluator {
    pub fn new(client: EvaluatorClient<Channel>, cfg: TransportConfig, batch_size: usize) -> Self {
        Self {
            client,
            cfg,
            batch_size,
        }
    }

    async fn eval_batch(
        client: &mut EvaluatorClient<Channel>,
        cfg: &TransportConfig,
        individuals: &[Vec<f64>],
    ) -> Result<Vec<f64>, Status> {
        let req = BatchRequest {
            individuals: individuals
                .iter()
                .map(|g| Individual { genes: g.clone() })
                .collect(),
        };

        let mut last_err: Option<Status> = None;
        for attempt in 0..cfg.max_attempts {
            let call = client.evaluate_batch(Request::new(req.clone()));
            match tokio::time::timeout(cfg.rpc_timeout, call).await {
                Ok(Ok(resp)) => {
                    return Ok(resp
                        .into_inner()
                        .results
                        .into_iter()
                        .map(|r| r.fitness)
                        .collect());
                }
                Ok(Err(e)) if e.code() == Code::Unavailable => {
                    info!("upstream not ready (attempt {}), retrying...", attempt + 1);
                    last_err = Some(e);
                    tokio::time::sleep(cfg.retry_delay).await;
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    info!(
                        "evaluate_batch timed out (attempt {}), retrying...",
                        attempt + 1
                    );
                    last_err = Some(Status::unavailable("rpc timed out"));
                    tokio::time::sleep(cfg.retry_delay).await;
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Status::unavailable("exhausted retries")))
    }
}

#[async_trait::async_trait]
impl Evaluator for GrpcEvaluator {
    async fn evaluate_population(&self, population: &[Vec<f64>]) -> Result<Vec<f64>, Status> {
        let chunks: Vec<&[Vec<f64>]> = population.chunks(self.batch_size).collect();
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            // Each task gets its own clone of the channel; tonic channels
            // are designed to be cloned cheaply for concurrency.
            let mut c = self.client.clone();
            let owned: Vec<Vec<f64>> = chunk.to_vec();
            let cfg = self.cfg.clone();
            handles.push(tokio::spawn(async move {
                Self::eval_batch(&mut c, &cfg, &owned).await
            }));
        }

        let mut flat = Vec::with_capacity(population.len());
        for h in handles {
            flat.extend(
                h.await
                    .map_err(|e| Status::internal(format!("join error: {e}")))??,
            );
        }
        Ok(flat)
    }
}
