pub mod grpc;
pub mod mock;
pub mod r#trait;

pub use grpc::GrpcSurrogateClient;
pub use mock::MockSurrogateClient;
pub use r#trait::SurrogateClient;
