use crate::proto::surrogate::surrogate_service_server::SurrogateService;
use crate::proto::surrogate::{
    BatchPredictRequest, BatchPredictResponse, IngestSamplesRequest, IngestSamplesResponse,
    PredictRequest, PredictResponse, StatusRequest, StatusResponse, TrainRequest, TrainResponse,
};
use crate::service::state::SurrogateState;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};

/// Single Responsibility: implements the gRPC transport layer for the
/// SurrogateService, delegating domain logic to `SurrogateState`.
pub struct SurrogateServer {
    state: Arc<SurrogateState>,
}

impl SurrogateServer {
    pub fn new(state: Arc<SurrogateState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl SurrogateService for SurrogateServer {
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        let genes = req.genes;

        if !self.state.is_ready().await {
            return Ok(Response::new(PredictResponse {
                fitness: 0.0,
                ready: false,
            }));
        }

        match self.state.predict_batch(&[genes]).await {
            Ok(fits) => {
                let fitness = fits.first().copied().unwrap_or(0.0);
                Ok(Response::new(PredictResponse {
                    fitness,
                    ready: true,
                }))
            }
            Err(e) => {
                warn!("Prediction error: {e}");
                Ok(Response::new(PredictResponse {
                    fitness: 0.0,
                    ready: false,
                }))
            }
        }
    }

    async fn predict_batch(
        &self,
        request: Request<BatchPredictRequest>,
    ) -> Result<Response<BatchPredictResponse>, Status> {
        let req = request.into_inner();
        let individuals: Vec<Vec<f64>> = req.individuals.into_iter().map(|ind| ind.genes).collect();

        if !self.state.is_ready().await {
            debug!("predict_batch called but surrogate model not yet ready");
            return Ok(Response::new(BatchPredictResponse {
                fitnesses: Vec::new(),
                ready: false,
            }));
        }

        match self.state.predict_batch(&individuals).await {
            Ok(fitnesses) => {
                debug!(
                    "predict_batch evaluated {} individuals successfully",
                    fitnesses.len()
                );
                Ok(Response::new(BatchPredictResponse {
                    fitnesses,
                    ready: true,
                }))
            }
            Err(e) => {
                warn!("predict_batch error: {e}");
                Ok(Response::new(BatchPredictResponse {
                    fitnesses: Vec::new(),
                    ready: false,
                }))
            }
        }
    }

    async fn ingest_samples(
        &self,
        request: Request<IngestSamplesRequest>,
    ) -> Result<Response<IngestSamplesResponse>, Status> {
        let req = request.into_inner();
        let raw_samples: Vec<(Vec<f64>, f64)> = req
            .samples
            .into_iter()
            .map(|s| (s.genes, s.fitness))
            .collect();

        let count = raw_samples.len();
        let (window_size, training_started) = self
            .state
            .ingest_samples(raw_samples, req.trigger_training)
            .await;

        info!(
            "Ingested {} true samples into sliding window (current window_size={})",
            count, window_size
        );

        Ok(Response::new(IngestSamplesResponse {
            current_window_size: window_size as u64,
            training_started,
        }))
    }

    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();
        let started = self.state.trigger_train_async(req.force).await;
        let window_len = self.state.window_size().await as u64;

        let metrics = self.state.last_metrics().await;
        let (r2, rmse) = metrics
            .map(|m| (m.r2_score, m.rmse))
            .unwrap_or((0.0, 0.0));

        Ok(Response::new(TrainResponse {
            success: started,
            samples_trained: window_len,
            r2_score: r2,
            rmse,
        }))
    }

    async fn get_status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let ready = self.state.is_ready().await;
        let window_size = self.state.window_size().await as u64;
        let max_window_size = self.state.config.window_size as u64;
        let model_version = self.state.model_version();
        let device_name = self.state.device_name().await;

        Ok(Response::new(StatusResponse {
            ready,
            window_size,
            max_window_size,
            model_version,
            device_name,
        }))
    }
}
