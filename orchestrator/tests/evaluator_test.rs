//! Integration tests for `GrpcEvaluator` against an in-process mock
//! `Evaluator` gRPC server. Verifies batching, ordering, and empty input.

use orchestrator::config::TransportConfig;
use orchestrator::evaluator::Evaluator;
use orchestrator::evaluator::GrpcEvaluator;
use orchestrator::proto::eval::evaluator_client::EvaluatorClient;
use orchestrator::proto::eval::evaluator_server::{
    Evaluator as GrpcEvaluatorService, EvaluatorServer,
};
use orchestrator::proto::eval::{BatchRequest, BatchResponse, EvaluationResult, Individual};
use std::time::Duration;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

/// Mock gRPC service: fitness = sum of genes. Deterministic, stateless.
#[derive(Default)]
struct SumEvalService;

#[tonic::async_trait]
impl GrpcEvaluatorService for SumEvalService {
    async fn evaluate(
        &self,
        req: Request<Individual>,
    ) -> Result<Response<EvaluationResult>, Status> {
        let ind = req.into_inner();
        let fitness = ind.genes.iter().sum::<f64>();
        Ok(Response::new(EvaluationResult {
            worker_id: "mock".into(),
            fitness,
        }))
    }

    async fn evaluate_batch(
        &self,
        req: Request<BatchRequest>,
    ) -> Result<Response<BatchResponse>, Status> {
        let results: Vec<EvaluationResult> = req
            .into_inner()
            .individuals
            .into_iter()
            .map(|ind| EvaluationResult {
                worker_id: "mock".into(),
                fitness: ind.genes.iter().sum(),
            })
            .collect();

        Ok(Response::new(BatchResponse { results }))
    }
}

/// Spawns a mock server on an ephemeral port and returns its address string.
async fn spawn_mock_server() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    tokio::spawn(async move {
        Server::builder()
            .add_service(EvaluatorServer::new(SumEvalService))
            .serve_with_incoming(incoming)
            .await
            .ok();
    });
    format!("http://{}", addr)
}

fn fast_transport() -> TransportConfig {
    TransportConfig {
        rpc_timeout: Duration::from_secs(5),
        max_attempts: 3,
        retry_delay: Duration::from_millis(10),
        channel_ready_deadline: Duration::from_secs(5),
        connect_timeout: Duration::from_secs(2),
        request_timeout: Duration::from_secs(5),
        keep_alive_timeout: Duration::from_secs(2),
        tcp_keepalive: Some(Duration::from_secs(5)),
    }
}

async fn connect_evaluator(endpoint: &str, batch_size: usize) -> GrpcEvaluator {
    let channel = tonic::transport::Channel::from_shared(endpoint.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let client = EvaluatorClient::new(channel);
    GrpcEvaluator::new(client, fast_transport(), batch_size)
}

#[tokio::test]
async fn evaluate_population_returns_correct_fitnesses() {
    let endpoint = spawn_mock_server().await;
    let eval = connect_evaluator(&endpoint, 2).await;
    let pop = vec![
        vec![1.0, 2.0, 3.0], // sum 6
        vec![0.5, 0.5],      // sum 1
        vec![10.0],          // sum 10
    ];
    let result = eval.evaluate_population(&pop).await.unwrap();
    assert_eq!(result.len(), pop.len());
    assert!((result[0] - 6.0).abs() < 1e-9);
    assert!((result[1] - 1.0).abs() < 1e-9);
    assert!((result[2] - 10.0).abs() < 1e-9);
}

#[tokio::test]
async fn evaluate_population_preserves_order_with_batching() {
    let endpoint = spawn_mock_server().await;
    // batch_size=1 forces one RPC per individual.
    let eval = connect_evaluator(&endpoint, 1).await;
    let pop: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
    let result = eval.evaluate_population(&pop).await.unwrap();
    assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[tokio::test]
async fn evaluate_population_empty_input_returns_empty() {
    let endpoint = spawn_mock_server().await;
    let eval = connect_evaluator(&endpoint, 4).await;
    let result = eval.evaluate_population(&[]).await.unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn evaluate_population_large_batch_handles_many_individuals() {
    let endpoint = spawn_mock_server().await;
    let eval = connect_evaluator(&endpoint, 7).await;
    let pop: Vec<Vec<f64>> = (0..50).map(|_| vec![1.0, 1.0]).collect();
    let result = eval.evaluate_population(&pop).await.unwrap();
    assert_eq!(result.len(), 50);
    for f in &result {
        assert!((f - 2.0).abs() < 1e-9);
    }
}
