use crate::surrogate_client::r#trait::SurrogateClient;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

/// Mock surrogate client for unit testing and offline simulation.
#[allow(clippy::type_complexity)]
pub struct MockSurrogateClient {
    pub is_ready_flag: AtomicBool,
    pub ingested: Arc<Mutex<Vec<(Vec<f64>, f64)>>>,
    pub default_prediction: f64,
}

impl MockSurrogateClient {
    pub fn new(ready: bool, default_prediction: f64) -> Self {
        Self {
            is_ready_flag: AtomicBool::new(ready),
            ingested: Arc::new(Mutex::new(Vec::new())),
            default_prediction,
        }
    }
}

#[async_trait::async_trait]
impl SurrogateClient for MockSurrogateClient {
    async fn predict_batch(&self, individuals: &[Vec<f64>]) -> Result<Option<Vec<f64>>, Status> {
        if !self.is_ready_flag.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(Some(vec![self.default_prediction; individuals.len()]))
    }

    async fn ingest_samples(&self, samples: &[(Vec<f64>, f64)]) -> Result<(), Status> {
        let mut guard = self.ingested.lock().await;
        guard.extend_from_slice(samples);
        Ok(())
    }

    async fn is_ready(&self) -> bool {
        self.is_ready_flag.load(Ordering::Relaxed)
    }
}
