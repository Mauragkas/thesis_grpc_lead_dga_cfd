//! SRP: thread-safe cached gRPC channel factory. Connects once per address.
//! Reusable across ring clients, migration services, and other transport adapters.

use std::collections::HashMap;
use tokio::sync::Mutex;
use tonic::transport::Channel;
use tonic::Status;

#[derive(Default, Debug)]
pub struct ChannelPool {
    channels: Mutex<HashMap<String, Channel>>,
}

impl ChannelPool {
    pub fn new() -> Self {
        Self {
            channels: Mutex::new(HashMap::new()),
        }
    }

    /// Returns a cached channel for `addr` or establishes a new connection.
    pub async fn get_or_connect(&self, addr: &str) -> Result<Channel, Status> {
        let mut cache = self.channels.lock().await;
        if let Some(ch) = cache.get(addr) {
            return Ok(ch.clone());
        }
        let uri = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr.to_string()
        } else {
            format!("http://{addr}")
        };
        let ch = Channel::from_shared(uri)
            .map_err(|e| Status::invalid_argument(format!("bad uri '{addr}': {e}")))?
            .connect()
            .await
            .map_err(|e| Status::unavailable(format!("connect '{addr}': {e}")))?;
        cache.insert(addr.to_string(), ch.clone());
        Ok(ch)
    }
}
