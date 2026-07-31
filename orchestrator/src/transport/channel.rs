use crate::config::TransportConfig;
use log::info;
use std::time::{Duration, Instant};
use tonic::transport::{Channel, Endpoint};
use tonic::Status;

/// SRP: builds a tonic `Endpoint` from a target string.
pub fn build_endpoint(
    target: &str,
    cfg: &TransportConfig,
) -> Result<Endpoint, Box<dyn std::error::Error>> {
    let uri = if target.starts_with("http://") || target.starts_with("https://") {
        target.to_string()
    } else {
        format!("http://{target}")
    };
    let endpoint = Channel::from_shared(uri)?
        .keep_alive_while_idle(true)
        .keep_alive_timeout(cfg.keep_alive_timeout)
        .tcp_keepalive(cfg.tcp_keepalive)
        .connect_timeout(cfg.connect_timeout)
        .timeout(cfg.request_timeout);
    Ok(endpoint)
}

/// SRP: blocks until a channel becomes ready or deadline elapses.
pub async fn wait_for_channel(endpoint: Endpoint, deadline: Duration) -> Result<Channel, Status> {
    let stop = Instant::now() + deadline;
    while Instant::now() < stop {
        match endpoint.clone().connect().await {
            Ok(ch) => return Ok(ch),
            Err(_) => {
                info!("Waiting for worker to become ready via Envoy...");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
    Err(Status::unavailable("Channel did not become ready in time."))
}
