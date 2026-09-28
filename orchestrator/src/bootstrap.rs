//! Composition root for the orchestrator service.
//!
//! Assembles all subsystems (ring, simulator, gene store, LEAD neighbor store,
//! surrogate node, migration, multi-tier evaluator) and executes the distributed GA run.

use crate::config::{
    config_from_env, GaConfig, LeadConfig, RingConfig, SurrogateClientConfig, TransportConfig,
};
use crate::evaluator::{GrpcEvaluator, MultiTierEvaluator};
use crate::ga::algorithm::GaRunner;
use crate::ga::telemetry::{JsonLinesFileSink, TelemetrySink};

use crate::gene_store::{EuclideanDistance, GenerationEvictor, InMemoryGeneStore};
use crate::hilbert::HilbertKeyGenerator;
use crate::lead_store::GrpcLeadStore;
use crate::migration::buffer::MigrantBuffer;
use crate::migration::lead_migration::LeadMigration;
use crate::migration::selector::TopKSelector;
use crate::neighbor_store::{HilbertNeighborStore, NeighborStore};
use crate::ring::{
    AddressHasher, GrpcRingClient, LocalRingMember, NodeInfo, RingServer, RingState, Sha256Hasher,
    Stabilizer,
};
use crate::surrogate_client::{GrpcSurrogateClient, SurrogateClient};
use crate::transport::channel::{build_endpoint, wait_for_channel};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tonic::transport::Server;
use tonic::Status;
use tracing::{error, info, warn};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Initializes tracing subscriber with dual logging:
/// - Stdout: Human-readable plain text for `docker logs`.
/// - File (if configured): Structured JSON format for Fluent-Bit.
pub fn init_logging() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // 1. Stdout layer: human-readable, plain text for `docker logs`
    let stdout_layer = tracing_subscriber::fmt::layer().with_target(true);

    // 2. Optional JSON file layer: for Fluent-Bit log forwarder
    let log_file_path = std::env::var("LOG_FILE_PATH").ok().or_else(|| {
        std::env::var("LOG_DIR").ok().map(|dir| {
            let name = std::env::var("CONTAINER_NAME")
                .or_else(|_| std::env::var("HOSTNAME"))
                .unwrap_or_else(|_| "orchestrator".into());
            format!("{dir}/{name}.log")
        })
    });

    if let Some(path_str) = log_file_path {
        let path = Path::new(&path_str);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(file) = OpenOptions::new().create(true).append(true).open(path) {
            let file_layer = tracing_subscriber::fmt::layer()
                .json()
                .with_target(true)
                .with_writer(Mutex::new(file));

            tracing_subscriber::registry()
                .with(env_filter)
                .with(stdout_layer)
                .with(file_layer)
                .init();
            return;
        }
    }

    tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer)
        .init();
}

/// Sets up and starts the ring gRPC server, client, and stabilizer.
pub async fn setup_ring(
    ring_cfg: &RingConfig,
    migrant_buffer: Arc<MigrantBuffer>,
) -> Result<(Arc<LocalRingMember>, Arc<GrpcRingClient>), Box<Status>> {
    let hasher = Sha256Hasher;
    let self_id = hasher.hash(&ring_cfg.self_address);
    let self_node = NodeInfo::new(self_id, ring_cfg.self_address.clone());
    let ring_state = Arc::new(RingState::new(self_node.clone()));
    let member = Arc::new(LocalRingMember::new(ring_state.clone()));

    let ring_server = RingServer::new(LocalRingMember::new(ring_state.clone()), migrant_buffer);

    let tonic_ring_server = crate::proto::ring::ring_server::RingServer::new(ring_server);

    let ring_bind = ring_cfg.bind_address.clone();
    let ring_addr: std::net::SocketAddr = ring_bind.parse().map_err(|e| {
        Box::new(Status::invalid_argument(format!(
            "bad RING_BIND '{ring_bind}': {e}"
        )))
    })?;
    info!("Ring gRPC server binding to {ring_addr}");
    tokio::spawn(async move {
        if let Err(e) = Server::builder()
            .add_service(tonic_ring_server)
            .serve(ring_addr)
            .await
        {
            error!("Ring gRPC server error: {e}");
        }
    });

    // Give the server a moment to bind.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let ring_client = Arc::new(GrpcRingClient::new());
    let stabilizer = Stabilizer::new(
        member.clone(),
        ring_client.clone(),
        ring_cfg.stabilize_interval,
        ring_cfg.successor_list_size,
    );

    if let Some(bootstrap) = &ring_cfg.bootstrap_address {
        for attempt in 1..=10 {
            match stabilizer.join(bootstrap).await {
                Ok(_) => break,
                Err(e) => {
                    warn!("Join attempt {attempt} failed: {e}; retrying in 2s...");
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
            }
        }
    } else {
        info!("No bootstrap; starting as the first ring member");
        member.set_successor(self_node.clone()).await;
    }

    stabilizer.spawn();
    Ok((member, ring_client))
}

/// Connects to the evaluation worker service and creates the gRPC simulator.
pub async fn setup_evaluator(
    ga_cfg: &GaConfig,
    transport_cfg: &TransportConfig,
) -> Result<GrpcEvaluator, Box<Status>> {
    let endpoint = build_endpoint(&ga_cfg.eval_endpoint, transport_cfg).map_err(|e| {
        error!(
            "Failed to build evaluator endpoint '{}': {e}",
            ga_cfg.eval_endpoint
        );
        e
    })?;
    info!("Waiting for evaluator endpoint to be ready...");
    let channel = wait_for_channel(endpoint, transport_cfg.channel_ready_deadline)
        .await
        .map_err(|e| {
            error!(
                "Evaluator channel not ready within {:?}: {e}",
                transport_cfg.channel_ready_deadline
            );
            Box::new(e)
        })?;
    info!("Evaluator channel ready");

    let client = crate::proto::eval::evaluator_client::EvaluatorClient::new(channel);
    Ok(GrpcEvaluator::new(
        client,
        transport_cfg.clone(),
        ga_cfg.batch_size,
    ))
}

/// Connects to the surrogate node microservice if configured.
pub async fn setup_surrogate_client(
    surrogate_cfg: &SurrogateClientConfig,
    transport_cfg: &TransportConfig,
) -> Result<Option<Arc<dyn SurrogateClient>>, Box<Status>> {
    if let Some(ep) = &surrogate_cfg.endpoint {
        let surrogate_endpoint = build_endpoint(ep, transport_cfg).map_err(|e| {
            error!("Failed to build surrogate endpoint '{ep}': {e}");
            e
        })?;
        info!("Waiting for surrogate node endpoint '{ep}' to be ready...");
        let channel = wait_for_channel(surrogate_endpoint, transport_cfg.channel_ready_deadline)
            .await
            .map_err(|e| {
                error!(
                    "Surrogate channel not ready within {:?}: {e}",
                    transport_cfg.channel_ready_deadline
                );
                Box::new(e)
            })?;
        info!("Surrogate node ready at {ep}");
        let client =
            crate::proto::surrogate::surrogate_service_client::SurrogateServiceClient::new(channel);
        let grpc_client = GrpcSurrogateClient::new(client, transport_cfg.clone());
        Ok(Some(Arc::new(grpc_client)))
    } else {
        info!("SURROGATE_ENDPOINT not set; surrogate tier disabled");
        Ok(None)
    }
}

/// Connects to the LEAD DHT node if configured and returns the neighbor store.
pub async fn setup_neighbor_store(
    lead_cfg: &LeadConfig,
    ga_cfg: &GaConfig,
    transport_cfg: &TransportConfig,
) -> Result<Option<Arc<HilbertNeighborStore<HilbertKeyGenerator, GrpcLeadStore>>>, Box<Status>> {
    if let Some(ep) = &lead_cfg.endpoint {
        let lead_endpoint = build_endpoint(ep, transport_cfg).map_err(|e| {
            error!("Failed to build LEAD endpoint '{ep}': {e}");
            e
        })?;
        info!("Waiting for LEAD node endpoint to be ready...");
        let lead_channel = wait_for_channel(lead_endpoint, transport_cfg.channel_ready_deadline)
            .await
            .map_err(|e| {
                error!(
                    "LEAD channel not ready within {:?}: {e}",
                    transport_cfg.channel_ready_deadline
                );
                Box::new(e)
            })?;
        info!("LEAD node ready at {ep}");
        let lead_client = crate::proto::lead::lead_client::LeadClient::new(lead_channel);
        let grpc_lead = GrpcLeadStore::new(lead_client);
        let keygen = HilbertKeyGenerator::new(ga_cfg.genes_len);
        Ok(Some(Arc::new(HilbertNeighborStore::new(keygen, grpc_lead))))
    } else {
        info!("LEAD_ENDPOINT not set; LEAD persistence disabled");
        Ok(None)
    }
}

/// Bootstraps all dependencies and runs the distributed GA algorithm to completion.
pub async fn run() -> Result<(), Box<Status>> {
    init_logging();

    let (
        ga_cfg,
        transport_cfg,
        store_cfg,
        lead_cfg,
        surrogate_cfg,
        tier_cfg,
        ring_cfg,
        migration_cfg,
    ) = config_from_env();

    info!(
        "GA config: pop_size={}, genes_len={}, max_generations={}, min_generations={}, stagnation_patience={}, min_improvement={}, mut_sigma={}, elite_frac={}, batch_size={}, seed={}",
        ga_cfg.pop_size,
        ga_cfg.genes_len,
        ga_cfg.max_generations,
        ga_cfg.min_generations,
        ga_cfg.stagnation_patience,
        ga_cfg.min_improvement,
        ga_cfg.mut_sigma,
        ga_cfg.elite_frac,
        ga_cfg.batch_size,
        ga_cfg.seed
    );
    info!("Evaluator endpoint: {}", ga_cfg.eval_endpoint);
    info!(
        "Tier config: ε_exact={}, R={}, k_neighbors={}",
        tier_cfg.epsilon_exact, tier_cfg.radius_r, tier_cfg.k_neighbors
    );
    info!(
        "Ring config: bind={}, self={}, bootstrap={:?}",
        ring_cfg.bind_address, ring_cfg.self_address, ring_cfg.bootstrap_address
    );
    info!(
        "Migration config: interval={} gens, count={}",
        migration_cfg.interval_generations, migration_cfg.migrant_count
    );

    let migrant_buffer = Arc::new(MigrantBuffer::new());
    let (member, ring_client) = setup_ring(&ring_cfg, migrant_buffer.clone()).await?;

    let raw_simulator = Arc::new(setup_evaluator(&ga_cfg, &transport_cfg).await?);

    let evictor = GenerationEvictor {
        max_age: store_cfg.max_age_generations,
    };
    let store = Arc::new(InMemoryGeneStore::new(EuclideanDistance, evictor));

    let neighbor_store = setup_neighbor_store(&lead_cfg, &ga_cfg, &transport_cfg).await?;
    let surrogate_client = setup_surrogate_client(&surrogate_cfg, &transport_cfg).await?;

    let multi_tier_evaluator = MultiTierEvaluator::new(
        raw_simulator.clone(),
        store.clone(),
        neighbor_store
            .clone()
            .map(|ns| ns as Arc<dyn NeighborStore>),
        surrogate_client,
        tier_cfg,
    );

    let migration = LeadMigration::new(
        migration_cfg,
        member.clone(),
        ring_client.clone(),
        TopKSelector,
        migrant_buffer.clone(),
    );

    let telemetry_sink: Option<Box<dyn TelemetrySink>> = match &ga_cfg.export_path {
        Some(path) => match JsonLinesFileSink::try_new(path) {
            Ok(sink) => Some(Box::new(sink)),
            Err(e) => {
                warn!("Failed to initialize GA telemetry export file at {path}: {e}");
                None
            }
        },
        None => None,
    };

    let mut rng = StdRng::seed_from_u64(ga_cfg.seed);
    let runner = GaRunner {
        cfg: &ga_cfg,
        evaluator: &multi_tier_evaluator,
        store: store.as_ref(),
        neighbor_store: neighbor_store.as_deref().map(|s| s as &dyn NeighborStore),
        migration: Some(&migration),
        telemetry: telemetry_sink.as_deref(),
    };


    match runner.run(&mut rng).await {
        Ok(res) => {
            info!(
                best_fitness = res.best_fitness,
                best_genome = ?res.best_genome,
                "GA run completed. Best fitness: {:.4}; best genome: {:?}",
                res.best_fitness,
                res.best_genome
            );
            Ok(())
        }
        Err(e) => {
            error!("GA run failed: {e}");
            Err(Box::new(e))
        }
    }
}
