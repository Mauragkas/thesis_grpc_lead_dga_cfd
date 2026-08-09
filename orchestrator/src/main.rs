use orchestrator::config::config_from_env;
use orchestrator::evaluator::{Evaluator, GrpcEvaluator};
use orchestrator::ga::algorithm::GaRunner;
use orchestrator::gene_store::{
    EuclideanDistance, GeneStore, GenerationEvictor, InMemoryGeneStore,
};
use orchestrator::hilbert::HilbertKeyGenerator;
use orchestrator::lead_store::GrpcLeadStore;
use orchestrator::neighbor_store::{HilbertNeighborStore, NeighborStore};
use orchestrator::transport::channel::{build_endpoint, wait_for_channel};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tonic::transport::Endpoint;
use tonic::Status;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Status> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .with_target(true)
        .init();

    let (ga_cfg, transport_cfg, store_cfg, lead_cfg) = config_from_env();
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

    info!("Starting aero GA run (hard volume constraint) via Envoy gRPC proxy...");

    let endpoint: Endpoint =
        build_endpoint(&ga_cfg.eval_endpoint, &transport_cfg).map_err(|e| {
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
    let store = InMemoryGeneStore::new(EuclideanDistance::default(), evictor);

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
        let chord_client = orchestrator::proto::chord::chord_client::ChordClient::new(lead_channel);
        let grpc_lead = GrpcLeadStore::new(chord_client);
        let keygen = HilbertKeyGenerator::new(ga_cfg.genes_len);
        Some(HilbertNeighborStore::new(keygen, grpc_lead))
    } else {
        info!("LEAD_ENDPOINT not set; LEAD persistence disabled");
        None
    };

    let mut rng = StdRng::seed_from_u64(ga_cfg.seed);
    let runner = GaRunner {
        cfg: &ga_cfg,
        evaluator: &evaluator as &dyn Evaluator,
        store: &store as &dyn GeneStore,
        neighbor_store: neighbor_store.as_ref().map(|s| s as &dyn NeighborStore),
    };
    info!("Starting GA run with seed {}", ga_cfg.seed);
    match runner.run(&mut rng).await {
        Ok(best) => {
            info!("GA run completed. Best fitness: {best:.4}");
            Ok(())
        }
        Err(e) => {
            error!("GA run failed: {e}");
            Err(e)
        }
    }
}
