//! Lead-style stabilizing ring for orchestrator nodes.
//!
//! Each orchestrator joins a ring, maintains successor/predecessor
//! pointers, and runs a background stabilization loop. The ring is
//! used to route migration traffic between GA islands.

pub mod client;
pub mod grpc_client;
pub mod grpc_server;
pub mod hash;
pub mod member;
pub mod stabilization;
pub mod state;

pub use client::RingClient;
pub use grpc_client::GrpcRingClient;
pub use grpc_server::RingServer;
pub use hash::{AddressHasher, Sha256Hasher};
pub use member::{LocalRingMember, RingMember};
pub use stabilization::Stabilizer;
pub use state::{NodeInfo, RingState};
