use orchestrator::config::config_from_env;
use orchestrator::evaluator::GrpcEvaluator;
use orchestrator::ga::algorithm::GaRunner;
use orchestrator::gene_store::{EuclideanDistance, GenerationEvictor, InMemoryGeneStore};
use orchestrator::hilbert::HilbertKeyGenerator;
use orchestrator::lead_store::GrpcLeadStore;
use orchestrator::migration::{LeadMigration, MigrantBuffer, TopKSelector};
use orchestrator::neighbor_store::{HilbertNeighborStore, NeighborStore};
use orchestrator::ring::{
    AddressHasher, GrpcRingClient, LocalRingMember, NodeInfo, RingServer, RingState, Sha256Hasher,
    Stabilizer,
};
use orchestrator::transport::channel::{build_endpoint, wait_for_channel};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Arc;
use tonic::transport::Server;
use tonic::Status;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<tonic::Status>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .with_target(true)
        .init();

    let (ga_cfg, transport_cfg, store_cfg, lead_cfg, ring_cfg, migration_cfg) = config_from_env();
    info!(
        "GA config: pop_size={}, genes_len={}, generations={}, mut_sigma={}, elite_frac={}, batch_size={}, seed={}",
        ga_cfg.pop_size,
        ga_cfg.genes_len,
        ga_cfg.generations,
        ga_cfg.mut_sigma,
        ga_cfg.elite_frac,
        ga_cfg.batch_size,
        ga_cfg.seed
    );
    info!("Evaluator endpoint: {}", ga_cfg.eval_endpoint);
    info!(
        "Ring config: bind={}, self={}, bootstrap={:?}",
        ring_cfg.bind_address, ring_cfg.self_address, ring_cfg.bootstrap_address
    );
    info!(
        "Migration config: interval={} gens, count={}",
        migration_cfg.interval_generations, migration_cfg.migrant_count
    );

    // --- Set up the ring ---
    let hasher = Sha256Hasher;
    let self_id = hasher.hash(&ring_cfg.self_address);
    let self_node = NodeInfo::new(self_id, ring_cfg.self_address.clone());
    let ring_state = Arc::new(RingState::new(self_node.clone()));
    let member = Arc::new(LocalRingMember::new(ring_state.clone()));

    let migrant_buffer = Arc::new(MigrantBuffer::new());
    let ring_server = RingServer::new(
        LocalRingMember::new(ring_state.clone()),
        migrant_buffer.clone(),
    );

    let tonic_ring_server = orchestrator::proto::ring::ring_server::RingServer::new(ring_server);

    let ring_bind = ring_cfg.bind_address.clone();
    let ring_addr: std::net::SocketAddr = ring_bind
        .parse()
        .map_err(|e| Status::invalid_argument(format!("bad RING_BIND '{ring_bind}': {e}")))?;
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

    // Set up the ring client + stabilizer.
    let ring_client = Arc::new(GrpcRingClient::new());
    let stabilizer = Stabilizer::new(
        member.clone(),
        ring_client.clone(),
        ring_cfg.stabilize_interval,
        ring_cfg.successor_list_size,
    );

    // Join the ring if we have a bootstrap node.
    if let Some(bootstrap) = &ring_cfg.bootstrap_address {
        // Retry join a few times — the bootstrap node may still be starting.
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
        // Successor is self.
        member.set_successor(self_node.clone()).await;
    }

    // Start the stabilization loop.
    stabilizer.spawn();

    // --- Set up the evaluator ---
    let endpoint = build_endpoint(&ga_cfg.eval_endpoint, &transport_cfg).map_err(|e| {
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
            e
        })?;
    info!("Evaluator channel ready");

    let client = orchestrator::proto::eval::evaluator_client::EvaluatorClient::new(channel);
    let evaluator = GrpcEvaluator::new(client, transport_cfg.clone(), ga_cfg.batch_size);

    let evictor = GenerationEvictor {
        max_age: store_cfg.max_age_generations,
    };
    let store = InMemoryGeneStore::new(EuclideanDistance, evictor);

    // --- Set up the LEAD neighbor store (optional) ---
    let neighbor_store = if let Some(ep) = &lead_cfg.endpoint {
        let lead_endpoint = build_endpoint(ep, &transport_cfg).map_err(|e| {
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
                e
            })?;
        info!("LEAD node ready at {ep}");
        let lead_client = orchestrator::proto::lead::lead_client::LeadClient::new(lead_channel);
        let grpc_lead = GrpcLeadStore::new(lead_client);
        let keygen = HilbertKeyGenerator::new(ga_cfg.genes_len);
        Some(HilbertNeighborStore::new(keygen, grpc_lead))
    } else {
        info!("LEAD_ENDPOINT not set; LEAD persistence disabled");
        None
    };

    // --- Set up migration ---
    let migration: LeadMigration<TopKSelector> = LeadMigration::new(
        migration_cfg,
        member.clone(),
        TopKSelector,
        migrant_buffer.clone(),
    );

    let mut rng = StdRng::seed_from_u64(ga_cfg.seed);
    let runner = GaRunner {
        cfg: &ga_cfg,
        evaluator: &evaluator,
        store: &store,
        neighbor_store: neighbor_store.as_ref().map(|s| s as &dyn NeighborStore),
        migration: Some(&migration),
    };
    info!("Starting GA run with seed {}", ga_cfg.seed);
    Ok(match runner.run(&mut rng).await {
        Ok(best) => {
            info!("GA run completed. Best fitness: {best:.4}");
            Ok(())
        }
        Err(e) => {
            error!("GA run failed: {e}");
            Err(e)
        }
    }?)
}
