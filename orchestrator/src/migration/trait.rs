//! ISP: the narrow contract the GA loop needs from migration.
//! OCP/DIP: new migration strategies (adaptive, topology-aware) implement
//! this trait without touching `GaRunner`.

use crate::migration::MigrantIndividual;
use tonic::Status;

#[async_trait::async_trait]
pub trait MigrationHook: Send + Sync {
    /// Called each generation. If migration is due this generation,
    /// selects and sends migrants to the ring successor.
    async fn maybe_emigrate(
        &self,
        generation: usize,
        population: &[Vec<f64>],
        fitnesses: &[f64],
    ) -> Result<(), Status>;

    /// Called each generation. Returns immigrants received since the
    /// last call (drains the buffer).
    async fn drain_immigrants(&self) -> Vec<MigrantIndividual>;
}
