#[tokio::main]
async fn main() -> Result<(), Box<tonic::Status>> {
    orchestrator::bootstrap::run().await
}

