//! Island-model migration for the distributed GA.
//!
//! Every `interval_generations` generations, each orchestrator sends
//! its top-K individuals to its ring successor. Incoming migrants
//! are buffered by the gRPC server and drained each generation by
//! the GA loop, replacing the worst individuals.

pub mod buffer;
pub mod config;
pub mod lead_migration;
pub mod selector;
pub mod r#trait;

pub use buffer::MigrantBuffer;
pub use config::MigrationConfig;
pub use lead_migration::LeadMigration;
pub use r#trait::MigrationHook;
pub use selector::{MigrantSelector, TopKSelector};

/// One individual received from (or selected for) migration.
#[derive(Debug, Clone)]
pub struct MigrantIndividual {
    pub genes: Vec<f64>,
    pub fitness: f64,
}

impl From<&MigrantIndividual> for crate::proto::ring::MigrantIndividual {
    fn from(m: &MigrantIndividual) -> Self {
        Self {
            genes: m.genes.clone(),
            fitness: m.fitness,
        }
    }
}

impl From<crate::proto::ring::MigrantIndividual> for MigrantIndividual {
    fn from(m: crate::proto::ring::MigrantIndividual) -> Self {
        Self {
            genes: m.genes,
            fitness: m.fitness,
        }
    }
}
