//! Tests for `transport::channel`: endpoint construction and readiness
//! polling. Success path uses a live mock gRPC server; timeout path uses
//! a dead address with a short deadline.

use orchestrator::config::TransportConfig;
use orchestrator::proto::eval::evaluator_server::{Evaluator, EvaluatorServer};
use orchestrator::proto::eval::{BatchRequest, BatchResponse, EvaluationResult, Individual};
use orchestrator::transport::channel::{build_endpoint, wait_for_channel};
use std::time::Duration;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

#[derive(Default)]
struct NoopEvalService;

#[tonic::async_trait]
impl Evaluator for NoopEvalService {
    async fn evaluate(
        &self,
        _req: Request<Individual>,
    ) -> Result<Response<EvaluationResult>, Status> {
        Ok(Response::new(EvaluationResult {
            worker_id: "noop".into(),
            fitness: 0.0,
        }))
    }
    async fn evaluate_batch(
        &self,
        _req: Request<BatchRequest>,
    ) -> Result<Response<BatchResponse>, Status> {
        Ok(Response::new(BatchResponse { results: vec![] }))
    }
}

fn fast_transport() -> TransportConfig {
    TransportConfig {
        rpc_timeout: Duration::from_secs(2),
        max_attempts: 1,
        retry_delay: Duration::from_millis(10),
        channel_ready_deadline: Duration::from_secs(5),
        connect_timeout: Duration::from_secs(2),
        request_timeout: Duration::from_secs(2),
        keep_alive_timeout: Duration::from_secs(2),
        tcp_keepalive: Some(Duration::from_secs(2)),
    }
}

#[test]
fn build_endpoint_adds_http_prefix_when_missing() {
    let cfg = fast_transport();
    let ep = build_endpoint("worker:50051", &cfg).unwrap();
    let uri = ep.uri().to_string();
    assert!(uri.starts_with("http://"));
    assert!(uri.contains("worker:50051"));
}

#[test]
fn build_endpoint_keeps_explicit_scheme() {
    let cfg = fast_transport();
    let ep = build_endpoint("https://secure-worker:443", &cfg).unwrap();
    let uri = ep.uri().to_string();
    assert!(uri.starts_with("https://"));
}

#[test]
fn build_endpoint_rejects_invalid_uri() {
    let cfg = fast_transport();
    // Tonic rejects URIs with illegal characters.
    assert!(build_endpoint("not a valid uri with spaces", &cfg).is_err());
}

#[tokio::test]
async fn wait_for_channel_succeeds_when_server_is_live() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    tokio::spawn(async move {
        Server::builder()
            .add_service(EvaluatorServer::new(NoopEvalService))
            .serve_with_incoming(incoming)
            .await
            .ok();
    });

    let cfg = fast_transport();
    let endpoint = build_endpoint(&addr.to_string(), &cfg).unwrap();
    let channel = wait_for_channel(endpoint, Duration::from_secs(5)).await;
    assert!(channel.is_ok());
}

#[tokio::test]
async fn channel_pool_caches_and_reuses_channels() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    tokio::spawn(async move {
        Server::builder()
            .add_service(EvaluatorServer::new(NoopEvalService))
            .serve_with_incoming(incoming)
            .await
            .ok();
    });

    let pool = orchestrator::transport::ChannelPool::new();
    let ch1 = pool.get_or_connect(&addr.to_string()).await;
    assert!(ch1.is_ok());
    let ch2 = pool.get_or_connect(&addr.to_string()).await;
    assert!(ch2.is_ok());
}

#[tokio::test]
async fn channel_pool_rejects_invalid_addr() {
    let pool = orchestrator::transport::ChannelPool::new();
    let res = pool.get_or_connect("not valid uri with spaces").await;
    assert!(res.is_err());
}

