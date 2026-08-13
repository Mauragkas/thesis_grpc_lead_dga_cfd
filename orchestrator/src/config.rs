use std::time::Duration;
use tracing::{info, warn};
use crate::migration::config::{MigrationConfig, RingConfig};

#[derive(Debug, Clone, Default)]
pub struct LeadConfig {
    pub endpoint: Option<String>,
}

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

/// Gene-store tunables. SRP: holds config only.
#[derive(Debug, Clone)]
pub struct GeneStoreConfig {
    /// How many generations a non-retrieved record may live before eviction.
    pub max_age_generations: usize,
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

impl Default for GeneStoreConfig {
    fn default() -> Self {
        Self {
            max_age_generations: 5,
        }
    }
}

/// Reads overrides from environment. SRP: parsing env, nothing else.
pub fn config_from_env() -> (
    GaConfig,
    TransportConfig,
    GeneStoreConfig,
    LeadConfig,
    RingConfig,
    MigrationConfig,
) {
    let mut ga = GaConfig::default();

    match std::env::var("EVAL_ENDPOINT") {
        Ok(endpoint) => {
            info!("EVAL_ENDPOINT override: {endpoint}");
            ga.eval_endpoint = endpoint;
        }
        Err(_) => info!(
            "EVAL_ENDPOINT not set; using default '{}'",
            ga.eval_endpoint
        ),
    }

    match std::env::var("GA_SEED") {
        Ok(raw) => match raw.parse::<u64>() {
            Ok(seed) => {
                info!("GA_SEED override: {seed}");
                ga.seed = seed;
            }
            Err(_) => warn!("Ignoring invalid GA_SEED '{raw}': not a valid u64"),
        },
        Err(_) => info!("GA_SEED not set; using default {}", ga.seed),
    }

    let mut store = GeneStoreConfig::default();
    match std::env::var("GENE_STORE_MAX_AGE") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(max_age) => {
                info!("GENE_STORE_MAX_AGE override: {max_age}");
                store.max_age_generations = max_age;
            }
            Err(_) => warn!("Ignoring invalid GENE_STORE_MAX_AGE '{raw}': not a valid usize"),
        },
        Err(_) => info!(
            "GENE_STORE_MAX_AGE not set; using default {}",
            store.max_age_generations
        ),
    }

    let mut lead = LeadConfig::default();
    match std::env::var("LEAD_ENDPOINT") {
        Ok(ep) => {
            info!("LEAD_ENDPOINT override: {ep}");
            lead.endpoint = Some(ep);
        }
        Err(_) => info!("LEAD_ENDPOINT not set; LEAD persistence disabled"),
    }

    let mut ring = RingConfig::default();
    match std::env::var("RING_BIND") {
        Ok(addr) => {
            info!("RING_BIND override: {addr}");
            ring.bind_address = addr;
        }
        Err(_) => info!("RING_BIND not set; using default '{}'", ring.bind_address),
    }
    match std::env::var("RING_SELF_ADDRESS") {
        Ok(addr) => {
            info!("RING_SELF_ADDRESS override: {addr}");
            ring.self_address = addr;
        }
        Err(_) => info!(
            "RING_SELF_ADDRESS not set; using default '{}'",
            ring.self_address
        ),
    }
    match std::env::var("RING_BOOTSTRAP") {
        Ok(addr) => {
            info!("RING_BOOTSTRAP override: {addr}");
            ring.bootstrap_address = Some(addr);
        }
        Err(_) => info!("RING_BOOTSTRAP not set; joining as first node"),
    }

    let mut migration = MigrationConfig::default();
    match std::env::var("MIGRATION_INTERVAL") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(n) => {
                info!("MIGRATION_INTERVAL override: {n}");
                migration.interval_generations = n;
            }
            Err(_) => warn!("Ignoring invalid MIGRATION_INTERVAL '{raw}'"),
        },
        Err(_) => info!(
            "MIGRATION_INTERVAL not set; using default {}",
            migration.interval_generations
        ),
    }
    match std::env::var("MIGRATION_COUNT") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(n) => {
                info!("MIGRATION_COUNT override: {n}");
                migration.migrant_count = n;
            }
            Err(_) => warn!("Ignoring invalid MIGRATION_COUNT '{raw}'"),
        },
        Err(_) => info!(
            "MIGRATION_COUNT not set; using default {}",
            migration.migrant_count
        ),
    }

    (ga, TransportConfig::default(), store, lead, ring, migration)
}

/// Clips a gene into the normalized [0,1] range. Pure helper.
pub fn clip(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}
