mod config;
mod evaluator;
mod ga;
mod proto;
mod transport;

use crate::config::config_from_env;
use crate::evaluator::{Evaluator, GrpcEvaluator};
use crate::ga::algorithm::GaRunner;
use crate::transport::channel::{build_endpoint, wait_for_channel};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tonic::transport::Endpoint;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let (ga_cfg, transport_cfg) = config_from_env();

    println!("Starting aero GA run (hard volume constraint) via Envoy gRPC proxy...");

    let endpoint: Endpoint = build_endpoint(&ga_cfg.eval_endpoint, &transport_cfg)?;
    info!("Waiting for evaluator endpoint to be ready...");
    let channel = wait_for_channel(endpoint, transport_cfg.channel_ready_deadline).await?;

    let client = crate::proto::eval::evaluator_client::EvaluatorClient::new(channel);
    let evaluator = GrpcEvaluator::new(client, transport_cfg.clone(), ga_cfg.batch_size);

    let mut rng = StdRng::seed_from_u64(ga_cfg.seed);
    let runner = GaRunner {
        cfg: &ga_cfg,
        evaluator: &evaluator as &dyn Evaluator,
    };
    runner.run(&mut rng).await?;
    Ok(())
}
