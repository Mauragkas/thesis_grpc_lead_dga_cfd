use std::time::Duration;

/// Single source of truth for tunable GA + transport parameters.
/// SRP: holds configuration only; no behaviour.
#[derive(Debug, Clone)]
pub struct GaConfig {
    pub pop_size: usize,
    pub genes_len: usize,
    pub generations: usize,
    pub mut_sigma: f64,
    pub elite_frac: f64,
    pub batch_size: usize,
    pub seed: u64,
    pub eval_endpoint: String,
}

/// Transport-level tunables, kept separate so transport code owns them.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub rpc_timeout: Duration,
    pub max_attempts: usize,
    pub retry_delay: Duration,
    pub channel_ready_deadline: Duration,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub keep_alive_timeout: Duration,
    pub tcp_keepalive: Option<Duration>,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            pop_size: 60,
            genes_len: 10,
            generations: 10,
            mut_sigma: 0.08,
            elite_frac: 0.5,
            batch_size: 1,
            seed: 42,
            eval_endpoint: "load-balancer:50051".to_string(),
        }
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            rpc_timeout: Duration::from_secs(120),
            max_attempts: 30,
            retry_delay: Duration::from_secs(2),
            channel_ready_deadline: Duration::from_secs(120),
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(300),
            keep_alive_timeout: Duration::from_secs(10),
            tcp_keepalive: Some(Duration::from_secs(30)),
        }
    }
}

/// Reads overrides from environment. SRP: parsing env, nothing else.
pub fn config_from_env() -> (GaConfig, TransportConfig) {
    let mut ga = GaConfig::default();
    if let Ok(endpoint) = std::env::var("EVAL_ENDPOINT") {
        ga.eval_endpoint = endpoint;
    }
    if let Some(seed) = std::env::var("GA_SEED").ok().and_then(|s| s.parse().ok()) {
        ga.seed = seed;
    }
    (ga, TransportConfig::default())
}

/// Clips a gene into the normalized [0,1] range. Pure helper.
pub fn clip(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}
