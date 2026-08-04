use log::info;
use orchestrator::config::config_from_env;
use orchestrator::evaluator::{Evaluator, GrpcEvaluator};
use orchestrator::ga::algorithm::GaRunner;
use orchestrator::gene_store::{
    EuclideanDistance, GeneStore, GenerationEvictor, InMemoryGeneStore,
};
use orchestrator::lead_store::{GrpcLeadStore, LeadStore};
use orchestrator::transport::channel::{build_endpoint, wait_for_channel};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tonic::transport::Endpoint;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let (ga_cfg, transport_cfg, store_cfg, lead_cfg) = config_from_env();

    println!("Starting aero GA run (hard volume constraint) via Envoy gRPC proxy...");

    let endpoint: Endpoint = build_endpoint(&ga_cfg.eval_endpoint, &transport_cfg)?;
    info!("Waiting for evaluator endpoint to be ready...");
    let channel = wait_for_channel(endpoint, transport_cfg.channel_ready_deadline).await?;

    let client = orchestrator::proto::eval::evaluator_client::EvaluatorClient::new(channel);
    let evaluator = GrpcEvaluator::new(client, transport_cfg.clone(), ga_cfg.batch_size);

    let evictor = GenerationEvictor {
        max_age: store_cfg.max_age_generations,
    };
    let store = InMemoryGeneStore::new(EuclideanDistance::default(), evictor);

    let lead_store = if let Some(ep) = &lead_cfg.endpoint {
        let lead_endpoint = build_endpoint(ep, &transport_cfg)?;
        info!("Waiting for LEAD node endpoint to be ready...");
        let lead_channel = wait_for_channel(lead_endpoint, transport_cfg.channel_ready_deadline).await?;
        let chord_client = orchestrator::proto::chord::chord_client::ChordClient::new(lead_channel);
        Some(GrpcLeadStore::new(chord_client))
    } else {
        None
    };

    let mut rng = StdRng::seed_from_u64(ga_cfg.seed);
    let runner = GaRunner {
        cfg: &ga_cfg,
        evaluator: &evaluator as &dyn Evaluator,
        store: &store as &dyn GeneStore,
        lead_store: lead_store.as_ref().map(|s| s as &dyn LeadStore),
    };
    runner.run(&mut rng).await?;
    Ok(())
}
