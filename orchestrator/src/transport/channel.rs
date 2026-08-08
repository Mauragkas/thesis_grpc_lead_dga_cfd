use crate::config::TransportConfig;
use std::time::{Duration, Instant};
use tonic::transport::{Channel, Endpoint};
use tonic::Status;
use tracing::{error, info, warn};

/// SRP: builds a tonic `Endpoint` from a target string.
pub fn build_endpoint(target: &str, cfg: &TransportConfig) -> Result<Endpoint, Status> {
    let uri = if target.starts_with("http://") || target.starts_with("https://") {
        target.to_string()
    } else {
        format!("http://{target}")
    };
    info!("Building endpoint for target '{target}' (uri '{uri}')");
    let endpoint = match Channel::from_shared(uri.clone()) {
        Ok(endpoint) => endpoint,
        Err(e) => {
            error!("Invalid endpoint URI '{uri}': {e}");
            return Err(Status::invalid_argument(format!(
                "Invalid endpoint URI '{uri}': {e}"
            )));
        }
    }
    .keep_alive_while_idle(true)
    .keep_alive_timeout(cfg.keep_alive_timeout)
    .tcp_keepalive(cfg.tcp_keepalive)
    .connect_timeout(cfg.connect_timeout)
    .timeout(cfg.request_timeout);
    Ok(endpoint)
}

/// SRP: blocks until a channel becomes ready or deadline elapses.
pub async fn wait_for_channel(endpoint: Endpoint, deadline: Duration) -> Result<Channel, Status> {
    let start = Instant::now();
    let stop = start + deadline;
    let mut attempt = 0u32;
    while Instant::now() < stop {
        attempt += 1;
        match endpoint.clone().connect().await {
            Ok(ch) => {
                info!(
                    "Channel ready after {:?} (attempt {attempt})",
                    start.elapsed()
                );
                return Ok(ch);
            }
            Err(e) => {
                warn!("Connect attempt {attempt} failed: {e}; retrying in 2s...");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
    error!("Channel did not become ready within {deadline:?} ({attempt} attempts)");
    Err(Status::unavailable("Channel did not become ready in time."))
}
