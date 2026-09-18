use std::time::Duration;
use tracing::{info, warn};
pub use crate::migration::config::{MigrationConfig, RingConfig};

#[derive(Debug, Clone, Default)]
pub struct LeadConfig {
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SurrogateClientConfig {
    pub endpoint: Option<String>,
}

/// Configuration for the Multi-Tier (ε-Bypass) evaluation pipeline.
#[derive(Debug, Clone)]
pub struct TierConfig {
    pub epsilon_exact: f64,
    pub radius_r: f64,
    pub k_neighbors: usize,
    pub min_neighbors: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            epsilon_exact: 0.005,
            radius_r: 0.15,
            k_neighbors: 15,
            min_neighbors: 1,
        }
    }
}

/// Single source of truth for tunable GA + transport parameters.
/// SRP: holds configuration only; no behaviour.
#[derive(Debug, Clone)]
pub struct GaConfig {
    pub pop_size: usize,
    pub genes_len: usize,
    /// Maximum generations to run (safety upper bound).
    pub max_generations: usize,
    /// Minimum generations to run before allowing early stopping.
    pub min_generations: usize,
    /// Number of consecutive generations without significant improvement (> min_improvement)
    /// before stopping. Set to 0 to disable early stopping.
    pub stagnation_patience: usize,
    /// Minimum absolute improvement in best fitness required to count as progress.
    pub min_improvement: f64,
    pub mut_sigma: f64,
    pub elite_frac: f64,
    pub batch_size: usize,
    pub seed: u64,
    pub eval_endpoint: String,
}

impl GaConfig {
    /// Convenience alias for `max_generations` for backward compatibility.
    pub fn generations(&self) -> usize {
        self.max_generations
    }
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
    pub circuit_breaker_failure_threshold: usize,
    pub circuit_breaker_recovery_timeout: Duration,
    pub circuit_breaker_half_open_probes: usize,
    pub max_individual_retries: usize,
    pub initial_retry_backoff: Duration,
    pub max_retry_backoff: Duration,
    pub retry_jitter_factor: f64,
    pub fallback_penalty_on_exhaustion: bool,
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
            max_generations: 100,
            min_generations: 10,
            stagnation_patience: 10,
            min_improvement: 0.001,
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
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_recovery_timeout: Duration::from_millis(1500),
            circuit_breaker_half_open_probes: 2,
            max_individual_retries: 5,
            initial_retry_backoff: Duration::from_millis(100),
            max_retry_backoff: Duration::from_millis(3000),
            retry_jitter_factor: 0.25,
            fallback_penalty_on_exhaustion: false,
        }
    }
}

use crate::gene_store::eviction::DEFAULT_MAX_AGE_GENERATIONS;
use std::collections::HashMap;

impl Default for GeneStoreConfig {
    fn default() -> Self {
        Self {
            max_age_generations: DEFAULT_MAX_AGE_GENERATIONS,
        }
    }
}

/// Reads overrides from an iterator of key-value pairs (e.g. environment variables or a map).
/// SRP: pure config parsing without hidden global state.
pub fn config_from_vars<I, K, V>(
    vars: I,
) -> (
    GaConfig,
    TransportConfig,
    GeneStoreConfig,
    LeadConfig,
    SurrogateClientConfig,
    TierConfig,
    RingConfig,
    MigrationConfig,
)
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let env_map: HashMap<String, String> = vars
        .into_iter()
        .map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string()))
        .collect();

    let mut ga = GaConfig::default();

    if let Some(endpoint) = env_map.get("EVAL_ENDPOINT") {
        info!("EVAL_ENDPOINT override: {endpoint}");
        ga.eval_endpoint = endpoint.clone();
    } else {
        info!(
            "EVAL_ENDPOINT not set; using default '{}'",
            ga.eval_endpoint
        );
    }

    if let Some(raw) = env_map.get("GA_SEED") {
        match raw.parse::<u64>() {
            Ok(seed) => {
                info!("GA_SEED override: {seed}");
                ga.seed = seed;
            }
            Err(_) => warn!("Ignoring invalid GA_SEED '{raw}': not a valid u64"),
        }
    } else {
        info!("GA_SEED not set; using default {}", ga.seed);
    }

    if let Some(raw) = env_map
        .get("MAX_GENERATIONS")
        .or_else(|| env_map.get("GA_MAX_GENERATIONS"))
        .or_else(|| env_map.get("GA_GENERATIONS"))
        .or_else(|| env_map.get("GENERATIONS"))
    {
        match raw.parse::<usize>() {
            Ok(gens) => {
                info!("MAX_GENERATIONS override: {gens}");
                ga.max_generations = gens;
            }
            Err(_) => warn!("Ignoring invalid MAX_GENERATIONS '{raw}': not a valid usize"),
        }
    } else {
        info!(
            "MAX_GENERATIONS not set; using default {}",
            ga.max_generations
        );
    }

    if let Some(raw) = env_map
        .get("MIN_GENERATIONS")
        .or_else(|| env_map.get("GA_MIN_GENERATIONS"))
    {
        match raw.parse::<usize>() {
            Ok(gens) => {
                info!("MIN_GENERATIONS override: {gens}");
                ga.min_generations = gens;
            }
            Err(_) => warn!("Ignoring invalid MIN_GENERATIONS '{raw}': not a valid usize"),
        }
    } else {
        info!(
            "MIN_GENERATIONS not set; using default {}",
            ga.min_generations
        );
    }

    if let Some(raw) = env_map
        .get("STAGNATION_PATIENCE")
        .or_else(|| env_map.get("GA_STAGNATION_PATIENCE"))
    {
        match raw.parse::<usize>() {
            Ok(patience) => {
                info!("STAGNATION_PATIENCE override: {patience}");
                ga.stagnation_patience = patience;
            }
            Err(_) => warn!("Ignoring invalid STAGNATION_PATIENCE '{raw}': not a valid usize"),
        }
    } else {
        info!(
            "STAGNATION_PATIENCE not set; using default {}",
            ga.stagnation_patience
        );
    }

    if let Some(raw) = env_map
        .get("MIN_IMPROVEMENT")
        .or_else(|| env_map.get("GA_MIN_IMPROVEMENT"))
    {
        match raw.parse::<f64>() {
            Ok(imp) => {
                info!("MIN_IMPROVEMENT override: {imp}");
                ga.min_improvement = imp;
            }
            Err(_) => warn!("Ignoring invalid MIN_IMPROVEMENT '{raw}': not a valid f64"),
        }
    } else {
        info!(
            "MIN_IMPROVEMENT not set; using default {}",
            ga.min_improvement
        );
    }

    let mut store = GeneStoreConfig::default();
    if let Some(raw) = env_map.get("GENE_STORE_MAX_AGE") {
        match raw.parse::<usize>() {
            Ok(max_age) => {
                info!("GENE_STORE_MAX_AGE override: {max_age}");
                store.max_age_generations = max_age;
            }
            Err(_) => warn!("Ignoring invalid GENE_STORE_MAX_AGE '{raw}': not a valid usize"),
        }
    } else {
        info!(
            "GENE_STORE_MAX_AGE not set; using default {}",
            store.max_age_generations
        );
    }

    let mut lead = LeadConfig::default();
    if let Some(ep) = env_map.get("LEAD_ENDPOINT") {
        info!("LEAD_ENDPOINT override: {ep}");
        lead.endpoint = Some(ep.clone());
    } else {
        info!("LEAD_ENDPOINT not set; LEAD persistence disabled");
    }

    let mut surrogate = SurrogateClientConfig::default();
    if let Some(ep) = env_map.get("SURROGATE_ENDPOINT") {
        info!("SURROGATE_ENDPOINT override: {ep}");
        surrogate.endpoint = Some(ep.clone());
    } else {
        info!("SURROGATE_ENDPOINT not set; surrogate tier disabled");
    }

    let mut tier = TierConfig::default();
    if let Some(raw) = env_map.get("TIER_EPSILON_EXACT") {
        if let Ok(eps) = raw.parse::<f64>() {
            tier.epsilon_exact = eps;
        }
    }
    if let Some(raw) = env_map.get("TIER_RADIUS_R") {
        if let Ok(r) = raw.parse::<f64>() {
            tier.radius_r = r;
        }
    }
    if let Some(raw) = env_map.get("TIER_K_NEIGHBORS") {
        if let Ok(k) = raw.parse::<usize>() {
            tier.k_neighbors = k;
        }
    }

    let mut ring = RingConfig::default();
    if let Some(addr) = env_map.get("RING_BIND") {
        info!("RING_BIND override: {addr}");
        ring.bind_address = addr.clone();
    } else {
        info!("RING_BIND not set; using default '{}'", ring.bind_address);
    }
    if let Some(addr) = env_map.get("RING_SELF_ADDRESS") {
        info!("RING_SELF_ADDRESS override: {addr}");
        ring.self_address = addr.clone();
    } else {
        info!(
            "RING_SELF_ADDRESS not set; using default '{}'",
            ring.self_address
        );
    }
    if let Some(addr) = env_map.get("RING_BOOTSTRAP") {
        info!("RING_BOOTSTRAP override: {addr}");
        ring.bootstrap_address = Some(addr.clone());
    } else {
        info!("RING_BOOTSTRAP not set; joining as first node");
    }

    let mut migration = MigrationConfig::default();
    if let Some(raw) = env_map.get("MIGRATION_INTERVAL") {
        match raw.parse::<usize>() {
            Ok(n) => {
                info!("MIGRATION_INTERVAL override: {n}");
                migration.interval_generations = n;
            }
            Err(_) => warn!("Ignoring invalid MIGRATION_INTERVAL '{raw}'"),
        }
    } else {
        info!(
            "MIGRATION_INTERVAL not set; using default {}",
            migration.interval_generations
        );
    }
    if let Some(raw) = env_map.get("MIGRATION_COUNT") {
        match raw.parse::<usize>() {
            Ok(n) => {
                info!("MIGRATION_COUNT override: {n}");
                migration.migrant_count = n;
            }
            Err(_) => warn!("Ignoring invalid MIGRATION_COUNT '{raw}'"),
        }
    } else {
        info!(
            "MIGRATION_COUNT not set; using default {}",
            migration.migrant_count
        );
    }

    let mut transport = TransportConfig::default();
    if let Some(raw) = env_map.get("CIRCUIT_BREAKER_FAILURE_THRESHOLD") {
        if let Ok(val) = raw.parse::<usize>() {
            info!("CIRCUIT_BREAKER_FAILURE_THRESHOLD override: {val}");
            transport.circuit_breaker_failure_threshold = val;
        }
    }
    if let Some(raw) = env_map.get("CIRCUIT_BREAKER_RECOVERY_TIMEOUT_MS") {
        if let Ok(ms) = raw.parse::<u64>() {
            info!("CIRCUIT_BREAKER_RECOVERY_TIMEOUT_MS override: {ms}ms");
            transport.circuit_breaker_recovery_timeout = Duration::from_millis(ms);
        }
    }
    if let Some(raw) = env_map.get("MAX_INDIVIDUAL_RETRIES") {
        if let Ok(val) = raw.parse::<usize>() {
            info!("MAX_INDIVIDUAL_RETRIES override: {val}");
            transport.max_individual_retries = val;
        }
    }
    if let Some(raw) = env_map.get("FALLBACK_PENALTY_ON_EXHAUSTION") {
        if let Ok(val) = raw.parse::<bool>() {
            info!("FALLBACK_PENALTY_ON_EXHAUSTION override: {val}");
            transport.fallback_penalty_on_exhaustion = val;
        }
    }

    (ga, transport, store, lead, surrogate, tier, ring, migration)
}

/// Reads overrides from process environment. Convenience wrapper around `config_from_vars`.
pub fn config_from_env() -> (
    GaConfig,
    TransportConfig,
    GeneStoreConfig,
    LeadConfig,
    SurrogateClientConfig,
    TierConfig,
    RingConfig,
    MigrationConfig,
) {
    config_from_vars(std::env::vars())
}

/// Clips a gene into the normalized [0,1] range. Pure helper.
pub fn clip(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}
