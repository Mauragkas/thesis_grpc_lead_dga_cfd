mod config;
mod evaluator;
mod ga;
mod gene_store;
mod proto;
mod transport;

use crate::config::config_from_env;
use crate::evaluator::{Evaluator, GrpcEvaluator};
use crate::ga::algorithm::GaRunner;
use crate::gene_store::{EuclideanDistance, GeneStore, GenerationEvictor, InMemoryGeneStore};
use crate::transport::channel::{build_endpoint, wait_for_channel};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tonic::transport::Endpoint;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let (ga_cfg, transport_cfg, store_cfg) = config_from_env();

    println!("Starting aero GA run (hard volume constraint) via Envoy gRPC proxy...");

    let endpoint: Endpoint = build_endpoint(&ga_cfg.eval_endpoint, &transport_cfg)?;
    info!("Waiting for evaluator endpoint to be ready...");
    let channel = wait_for_channel(endpoint, transport_cfg.channel_ready_deadline).await?;

    let client = crate::proto::eval::evaluator_client::EvaluatorClient::new(channel);
    let evaluator = GrpcEvaluator::new(client, transport_cfg.clone(), ga_cfg.batch_size);

    // DIP: wire a concrete store behind the `GeneStore` trait.
    let evictor = GenerationEvictor {
        max_age: store_cfg.max_age_generations,
    };
    let store = InMemoryGeneStore::new(EuclideanDistance::default(), evictor);

    let mut rng = StdRng::seed_from_u64(ga_cfg.seed);
    let runner = GaRunner {
        cfg: &ga_cfg,
        evaluator: &evaluator as &dyn Evaluator,
        store: &store as &dyn GeneStore,
    };
    runner.run(&mut rng).await?;
    Ok(())
}
