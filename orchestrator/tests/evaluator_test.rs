//! Integration tests for `GrpcEvaluator` against an in-process mock
//! `Evaluator` gRPC server. Verifies batching, ordering, empty input,
//! circuit breaking, and per-individual retry resiliency.

use orchestrator::config::TransportConfig;
use orchestrator::evaluator::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
use orchestrator::evaluator::Evaluator;
use orchestrator::evaluator::GrpcEvaluator;
use orchestrator::proto::eval::evaluator_client::EvaluatorClient;
use orchestrator::proto::eval::evaluator_server::{
    Evaluator as GrpcEvaluatorService, EvaluatorServer,
};
use orchestrator::proto::eval::{BatchRequest, BatchResponse, EvaluationResult, Individual};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
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

/// Mock service that fails initial calls with `Unavailable` to simulate
/// worker dropping before DNS refresh.
struct FlakyEvalService {
    fail_count: Arc<AtomicUsize>,
    batch_fail_always: bool,
}

impl FlakyEvalService {
    fn should_fail(&self) -> bool {
        self.fail_count
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |val| {
                if val > 0 {
                    Some(val - 1)
                } else {
                    None
                }
            })
            .is_ok()
    }
}

#[tonic::async_trait]
impl GrpcEvaluatorService for FlakyEvalService {
    async fn evaluate(
        &self,
        req: Request<Individual>,
    ) -> Result<Response<EvaluationResult>, Status> {
        if self.should_fail() {
            return Err(Status::unavailable("simulated upstream drop before DNS refresh"));
        }
        let ind = req.into_inner();
        let fitness = ind.genes.iter().sum::<f64>();
        Ok(Response::new(EvaluationResult {
            worker_id: "flaky-recovered".into(),
            fitness,
        }))
    }

    async fn evaluate_batch(
        &self,
        req: Request<BatchRequest>,
    ) -> Result<Response<BatchResponse>, Status> {
        if self.batch_fail_always || self.should_fail() {
            return Err(Status::unavailable("simulated upstream drop before DNS refresh"));
        }
        let results: Vec<EvaluationResult> = req
            .into_inner()
            .individuals
            .into_iter()
            .map(|ind| EvaluationResult {
                worker_id: "flaky-recovered".into(),
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

async fn spawn_flaky_server(initial_failures: usize, batch_fail_always: bool) -> (String, Arc<AtomicUsize>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let fail_count = Arc::new(AtomicUsize::new(initial_failures));
    let service = FlakyEvalService {
        fail_count: fail_count.clone(),
        batch_fail_always,
    };
    tokio::spawn(async move {
        Server::builder()
            .add_service(EvaluatorServer::new(service))
            .serve_with_incoming(incoming)
            .await
            .ok();
    });
    (format!("http://{}", addr), fail_count)
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
        circuit_breaker_failure_threshold: 3,
        circuit_breaker_recovery_timeout: Duration::from_millis(100),
        circuit_breaker_half_open_probes: 1,
        max_individual_retries: 5,
        initial_retry_backoff: Duration::from_millis(10),
        max_retry_backoff: Duration::from_millis(50),
        retry_jitter_factor: 0.1,
        fallback_penalty_on_exhaustion: false,
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

#[tokio::test]
async fn circuit_breaker_state_transitions() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 2,
        recovery_timeout: Duration::from_millis(50),
        half_open_probes: 2,
    });

    assert_eq!(cb.state(), CircuitState::Closed);
    assert!(cb.can_execute());

    // 1 failure: still closed
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);
    assert!(cb.can_execute());

    // 2nd failure: trips to Open
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
    assert!(!cb.can_execute());

    // Wait for recovery timeout
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert_eq!(cb.state(), CircuitState::HalfOpen);
    assert!(cb.can_execute());

    // Successful probes in HalfOpen recover to Closed
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::HalfOpen);
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
    assert!(cb.can_execute());
}

#[tokio::test]
async fn evaluator_recovers_from_transient_upstream_unavailable() {
    // Fails first 2 calls, then recovers
    let (endpoint, _) = spawn_flaky_server(2, false).await;
    let eval = connect_evaluator(&endpoint, 1).await;

    let pop = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let result = eval.evaluate_population(&pop).await.unwrap();
    assert_eq!(result, vec![3.0, 7.0]);
}

#[tokio::test]
async fn evaluator_batch_fallback_to_individual_evaluations() {
    // Batch requests fail always, but individual requests succeed after initial failure
    let (endpoint, _) = spawn_flaky_server(1, true).await;
    let eval = connect_evaluator(&endpoint, 2).await;

    let pop = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
    let result = eval.evaluate_population(&pop).await.unwrap();
    assert_eq!(result, vec![2.0, 4.0]);
}

#[tokio::test]
async fn evaluator_fallback_penalty_on_exhaustion() {
    // 100 failures -> retries will be exhausted
    let (endpoint, _) = spawn_flaky_server(100, false).await;
    let channel = tonic::transport::Channel::from_shared(endpoint)
        .unwrap()
        .connect()
        .await
        .unwrap();
    let client = EvaluatorClient::new(channel);

    let mut cfg = fast_transport();
    cfg.max_individual_retries = 2;
    cfg.fallback_penalty_on_exhaustion = true;

    let eval = GrpcEvaluator::new(client, cfg, 1);
    let pop = vec![vec![1.0, 2.0]];
    let result = eval.evaluate_population(&pop).await.unwrap();
    assert_eq!(result, vec![-1e9]);
}
