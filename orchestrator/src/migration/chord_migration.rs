//! SRP: orchestrates migration using the ring successor as the
//! migration target. Depends on `RingClient` (DIP), `MigrantSelector`
//! (OCP), and `MigrantBuffer` for incoming individuals.

use crate::migration::buffer::MigrantBuffer;
use crate::migration::config::MigrationConfig;
use crate::migration::selector::MigrantSelector;
use crate::migration::{MigrantIndividual, MigrationHook};
use crate::proto::ring::ring_client::RingClient as GrpcRingClientProto;
use crate::proto::ring::{MigrantIndividual as ProtoMigrant, MigrateRequest};
use crate::ring::member::LocalRingMember;
use std::sync::Arc;
use tonic::transport::Channel;
use tonic::Status;
use tracing::{debug, info, warn};

pub struct LeadMigration<S: MigrantSelector> {
    config: MigrationConfig,
    member: Arc<LocalRingMember>,
    selector: S,
    buffer: Arc<MigrantBuffer>,
    /// Channel cache for sending migrants to successors.
    channels: tokio::sync::Mutex<std::collections::HashMap<String, Channel>>,
}

impl<S: MigrantSelector> LeadMigration<S> {
    pub fn new(
        config: MigrationConfig,
        member: Arc<LocalRingMember>,
        selector: S,
        buffer: Arc<MigrantBuffer>,
    ) -> Self {
        Self {
            config,
            member,
            selector,
            buffer,
            channels: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    async fn channel_for(&self, addr: &str) -> Result<Channel, Status> {
        let mut cache = self.channels.lock().await;
        if let Some(ch) = cache.get(addr) {
            return Ok(ch.clone());
        }
        let uri = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr.to_string()
        } else {
            format!("http://{addr}")
        };
        let ch = Channel::from_shared(uri)
            .map_err(|e| Status::invalid_argument(format!("bad uri '{addr}': {e}")))?
            .connect()
            .await
            .map_err(|e| Status::unavailable(format!("connect '{addr}': {e}")))?;
        cache.insert(addr.to_string(), ch.clone());
        Ok(ch)
    }
}

#[async_trait::async_trait]
impl<S: MigrantSelector> MigrationHook for LeadMigration<S> {
    async fn maybe_emigrate(
        &self,
        generation: usize,
        population: &[Vec<f64>],
        fitnesses: &[f64],
    ) -> Result<(), Status> {
        if generation == 0 || generation % self.config.interval_generations != 0 {
            return Ok(());
        }

        let migrants = self
            .selector
            .select(population, fitnesses, self.config.migrant_count);

        if migrants.is_empty() {
            return Ok(());
        }

        let succ = self.member.state().successor.lock().await.clone();
        let succ = match succ {
            Some(s) if s.id != self.member.state().self_node.id => s,
            _ => {
                debug!("No distinct successor; skipping migration at gen {generation}");
                return Ok(());
            }
        };

        let proto_migrants: Vec<ProtoMigrant> = migrants
            .iter()
            .map(|m| ProtoMigrant {
                genes: m.genes.clone(),
                fitness: m.fitness,
            })
            .collect();

        info!(
            "Migrating {} individuals to successor {} ({}) at gen {generation}",
            proto_migrants.len(),
            succ.id,
            succ.address
        );

        let ch = self.channel_for(&succ.address).await?;
        let mut client = GrpcRingClientProto::new(ch);
        match client
            .migrate(MigrateRequest {
                individuals: proto_migrants,
                sender: self.member.state().self_node.address.clone(),
            })
            .await
        {
            Ok(resp) => {
                if resp.into_inner().accepted {
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
