//! SRP: orchestrates migration using the ring successor as the
//! migration target. Depends on `RingMember` and `RingClient` (DIP), `MigrantSelector`
//! (OCP), and `MigrantBuffer` for incoming individuals.

use crate::migration::buffer::MigrantBuffer;
use crate::migration::config::MigrationConfig;
use crate::migration::selector::MigrantSelector;
use crate::migration::{MigrantIndividual, MigrationHook};
use crate::ring::client::RingClient;
use crate::ring::member::RingMember;
use std::sync::Arc;
use tonic::Status;
use tracing::{debug, info, warn};

pub struct LeadMigration<S: MigrantSelector, M: RingMember, C: RingClient> {
    config: MigrationConfig,
    member: Arc<M>,
    client: Arc<C>,
    selector: S,
    buffer: Arc<MigrantBuffer>,
}

impl<S: MigrantSelector, M: RingMember, C: RingClient> LeadMigration<S, M, C> {
    pub fn new(
        config: MigrationConfig,
        member: Arc<M>,
        client: Arc<C>,
        selector: S,
        buffer: Arc<MigrantBuffer>,
    ) -> Self {
        Self {
            config,
            member,
            client,
            selector,
            buffer,
        }
    }
}

#[async_trait::async_trait]
impl<S: MigrantSelector, M: RingMember, C: RingClient> MigrationHook
    for LeadMigration<S, M, C>
{
    async fn maybe_emigrate(
        &self,
        generation: usize,
        population: &[Vec<f64>],
        fitnesses: &[f64],
    ) -> Result<(), Status> {
        if generation == 0 || !generation.is_multiple_of(self.config.interval_generations) {
            return Ok(());
        }

        let migrants = self
            .selector
            .select(population, fitnesses, self.config.migrant_count);

        if migrants.is_empty() {
            return Ok(());
        }

        let self_node = self.member.self_node();
        let succ = self.member.get_successor().await;
        let succ = match succ {
            Some(s) if s.id != self_node.id => s,
            _ => {
                debug!("No distinct successor; skipping migration at gen {generation}");
                return Ok(());
            }
        };

        info!(
            "Migrating {} individuals to successor {} ({}) at gen {generation}",
            migrants.len(),
            succ.id,
            succ.address
        );

        match self
            .client
            .migrate(&succ.address, &self_node.address, &migrants)
            .await
        {
            Ok(accepted) => {
                if accepted {
                    debug!("Migration accepted by {}", succ.address);
                } else {
                    warn!("Migration rejected by {}", succ.address);
                }
            }
            Err(e) => {
                warn!("Migration to {} failed: {e}", succ.address);
                // Don't propagate — migration is best-effort.
            }
        }

        Ok(())
    }

    async fn drain_immigrants(&self) -> Vec<MigrantIndividual> {
        self.buffer.drain().await
    }
}
