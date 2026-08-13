//! SRP: runs the Chord stabilization loop. Periodically verifies the
//! successor pointer and notifies the successor. Depends on `RingClient`
//! (DIP) and `RingState` — never on concrete gRPC types.

use crate::ring::client::RingClient;
use crate::ring::member::LocalRingMember;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, info, warn};

/// Runs stabilization + successor-list refresh in a background task.
pub struct Stabilizer<C: RingClient> {
    member: Arc<LocalRingMember>,
    client: Arc<C>,
    interval: Duration,
    successor_list_size: usize,
}

impl<C: RingClient + 'static> Stabilizer<C> {
    pub fn new(
        member: Arc<LocalRingMember>,
        client: Arc<C>,
        interval: Duration,
        successor_list_size: usize,
    ) -> Self {
        Self {
            member,
            client,
            interval,
            successor_list_size,
        }
    }

    /// Spawn the stabilization loop as a detached tokio task.
    pub fn spawn(self) {
        tokio::spawn(async move {
            self.run().await;
        });
    }

    async fn run(&self) {
        let mut ticker = interval(self.interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // Skip the immediate first tick.
        ticker.tick().await;

        loop {
            ticker.tick().await;
            if let Err(e) = self.stabilize().await {
                warn!("Stabilization iteration failed: {e}");
            }
            if let Err(e) = self.refresh_successor_list().await {
                warn!("Successor-list refresh failed: {e}");
            }
        }
    }

    /// Core Chord stabilization:
    /// 1. Ask successor for its predecessor `x`.
    /// 2. If `x` is in (self, successor), set `x` as our successor.
    /// 3. Notify our successor about us.
    async fn stabilize(&self) -> Result<(), tonic::Status> {
        let self_node = self.member.state().self_node.clone();
        let succ = {
            let s = self.member.state().successor.lock().await.clone();
            s
        };

        let succ = match succ {
            Some(s) if s.id != self_node.id => s,
            _ => {
                // No successor or only self — nothing to stabilize.
                return Ok(());
            }
        };

        // Ask successor for its predecessor.
        let x = self.client.get_predecessor(&succ.address).await?;

        if let Some(x) = x {
            if crate::ring::state::RingState::in_open(self_node.id, succ.id, x.id) {
                debug!("Stabilize: updating successor to {} ({})", x.id, x.address);
                self.member.set_successor(x.clone()).await;
                // Notify the new successor.
                let _ = self.client.notify(&x.address, self_node.clone()).await;
                return Ok(());
            }
        }

        // Notify current successor.
        let _ = self.client.notify(&succ.address, self_node.clone()).await;
        Ok(())
    }

    /// Refresh the successor list by asking our successor for its list
    /// and prepending our successor.
    async fn refresh_successor_list(&self) -> Result<(), tonic::Status> {
        let succ = self.member.state().successor.lock().await.clone();
        let succ = match succ {
            Some(s) if s.id != self.member.state().self_node.id => s,
            _ => {
                // Single-node ring: list is just us.
                self.member
                    .set_successor_list(vec![self.member.state().self_node.clone()])
                    .await;
                return Ok(());
            }
        };

        let mut list = vec![succ.clone()];
        match self.client.get_successor_list(&succ.address).await {
            Ok(remote) => {
                for n in remote {
                    if list.len() >= self.successor_list_size {
                        break;
                    }
                    if n.id == self.member.state().self_node.id {
                        break;
                    }
                    list.push(n);
                }
            }
            Err(e) => {
                warn!("Failed to fetch successor list from {}: {e}", succ.address);
            }
        }

        self.member.set_successor_list(list).await;
        Ok(())
    }

    /// Join an existing ring via a bootstrap node.
    /// Asks bootstrap to find our successor, then sets it.
    pub async fn join(&self, bootstrap_addr: &str) -> Result<(), tonic::Status> {
        let self_node = self.member.state().self_node.clone();
        info!(
            "Joining ring via bootstrap '{bootstrap_addr}' (self id={})",
            self_node.id
        );
        let succ = self
            .client
            .find_successor(bootstrap_addr, self_node.id)
            .await?;
        info!("Join: successor is {} ({})", succ.id, succ.address);
        self.member.set_successor(succ).await;
        Ok(())
    }
}
