# Transport Module

Tonic transport configuration, endpoint construction, and channel readiness polling.

## Files and Code Structure

### 1. `channel.rs`

- **`pub fn build_endpoint(target: &str, cfg: &TransportConfig) -> Result<Endpoint, tonic::Status>`**:
  - Automatically prefixes target with `http://` if no scheme is provided.
  - Configures tonic `Endpoint` with:
    - `keep_alive_while_idle(true)`
    - `keep_alive_timeout(cfg.keep_alive_timeout)`
    - `tcp_keepalive(cfg.tcp_keepalive)`
    - `connect_timeout(cfg.connect_timeout)`
    - `timeout(cfg.request_timeout)`
  - Returns configured `tonic::transport::Endpoint`.

- **`pub async fn wait_for_channel(endpoint: Endpoint, deadline: Duration) -> Result<Channel, tonic::Status>`**:
  - Computes stop instant `Instant::now() + deadline`.
  - Loops attempting `endpoint.clone().connect().await`.
  - Returns connected `tonic::transport::Channel` on success.
  - Sleeps 2 seconds between failed attempts.
  - Returns `Status::unavailable` if deadline is exceeded.
