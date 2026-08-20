use crate::config::TransportConfig;
use crate::proto::surrogate::surrogate_service_client::SurrogateServiceClient;
use crate::proto::surrogate::{
    BatchPredictRequest, Individual, IngestSamplesRequest, Sample,
};
use crate::surrogate_client::r#trait::SurrogateClient;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tonic::transport::Channel;
use tonic::{Request, Status};
use tracing::{debug, error, warn};

/// Single Responsibility: concrete gRPC-backed surrogate client.
pub struct GrpcSurrogateClient {
    client: SurrogateServiceClient<Channel>,
    cfg: TransportConfig,
    is_ready_cache: Arc<AtomicBool>,
}

impl GrpcSurrogateClient {
    pub fn new(client: SurrogateServiceClient<Channel>, cfg: TransportConfig) -> Self {
        Self {
            client,
            cfg,
            is_ready_cache: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait::async_trait]
impl SurrogateClient for GrpcSurrogateClient {
    async fn predict_batch(&self, individuals: &[Vec<f64>]) -> Result<Option<Vec<f64>>, Status> {
        if individuals.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let mut client = self.client.clone();
        let req = BatchPredictRequest {
            individuals: individuals
                .iter()
                .map(|g| Individual { genes: g.clone() })
                .collect(),
        };

        let call = client.predict_batch(Request::new(req));
        match tokio::time::timeout(self.cfg.rpc_timeout, call).await {
            Ok(Ok(resp)) => {
                let inner = resp.into_inner();
                self.is_ready_cache.store(inner.ready, Ordering::Relaxed);
                if inner.ready {
                    debug!(
                        "Surrogate evaluated batch of {} individuals",
                        inner.fitnesses.len()
                    );
                    Ok(Some(inner.fitnesses))
                } else {
                    debug!("Surrogate reported model not ready (cold-start)");
                    Ok(None)
                }
            }
            Ok(Err(e)) => {
                warn!("Surrogate gRPC predict_batch error: {e}");
                self.is_ready_cache.store(false, Ordering::Relaxed);
                Err(e)
            }
            Err(_) => {
                warn!("Surrogate predict_batch timed out after {:?}", self.cfg.rpc_timeout);
                self.is_ready_cache.store(false, Ordering::Relaxed);
                Err(Status::deadline_exceeded("surrogate timeout"))
            }
        }
    }

    async fn ingest_samples(&self, samples: &[(Vec<f64>, f64)]) -> Result<(), Status> {
        if samples.is_empty() {
            return Ok(());
        }

        let mut client = self.client.clone();
        let req = IngestSamplesRequest {
            samples: samples
                .iter()
                .map(|(g, f)| Sample {
                    genes: g.clone(),
                    fitness: *f,
                })
                .collect(),
            trigger_training: false,
        };

        let call = client.ingest_samples(Request::new(req));
        match tokio::time::timeout(self.cfg.rpc_timeout, call).await {
            Ok(Ok(resp)) => {
                let inner = resp.into_inner();
                debug!(
                    "Ingested {} samples into surrogate (window_size={}, training_started={})",
                    samples.len(),
                    inner.current_window_size,
                    inner.training_started
                );
                Ok(())
            }
            Ok(Err(e)) => {
                error!("Surrogate ingest_samples gRPC error: {e}");
                Err(e)
            }
            Err(_) => {
                warn!("Surrogate ingest_samples timed out");
                Err(Status::deadline_exceeded("surrogate timeout"))
            }
        }
    }

    async fn is_ready(&self) -> bool {
        self.is_ready_cache.load(Ordering::Relaxed)
    }
}
