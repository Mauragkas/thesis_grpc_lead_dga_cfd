use orchestrator::migration::buffer::MigrantBuffer;
use orchestrator::migration::config::MigrationConfig;
use orchestrator::migration::{LeadMigration, MigrantIndividual, MigrantSelector, MigrationHook, TopKSelector};
use orchestrator::ring::client::RingClient;
use orchestrator::ring::member::RingMember;
use orchestrator::ring::state::NodeInfo;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

#[test]
fn top_k_selects_highest_fitness() {
    let pop = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let fit = vec![10.0, 30.0, 20.0, 40.0];
    let migrants = TopKSelector.select(&pop, &fit, 2);
    assert_eq!(migrants.len(), 2);
    assert_eq!(migrants[0].genes, vec![3.0]); // fitness 40
    assert_eq!(migrants[1].genes, vec![1.0]); // fitness 30
}

#[test]
fn migrant_individual_proto_conversion_roundtrip() {
    let ind = MigrantIndividual {
        genes: vec![0.1, 0.2, 0.3],
        fitness: 42.5,
    };
    let proto = orchestrator::proto::ring::MigrantIndividual::from(&ind);
    assert_eq!(proto.genes, vec![0.1, 0.2, 0.3]);
    assert_eq!(proto.fitness, 42.5);

    let back = MigrantIndividual::from(proto);
    assert_eq!(back.genes, ind.genes);
    assert_eq!(back.fitness, ind.fitness);
}

struct MockRingMember {
    self_node: NodeInfo,
    successor: Mutex<Option<NodeInfo>>,
}

#[async_trait::async_trait]
impl RingMember for MockRingMember {
    async fn find_successor(&self, _id: u64) -> NodeInfo {
        self.self_node.clone()
    }
    async fn get_predecessor(&self) -> Option<NodeInfo> {
        None
    }
    async fn get_successor(&self) -> Option<NodeInfo> {
        self.successor.lock().await.clone()
    }
    fn self_node(&self) -> NodeInfo {
        self.self_node.clone()
    }
    async fn notify(&self, _other: NodeInfo) -> bool {
        true
    }
    async fn get_successor_list(&self) -> Vec<NodeInfo> {
        vec![]
    }
    async fn set_successor(&self, node: NodeInfo) {
        *self.successor.lock().await = Some(node);
    }
    async fn set_successor_list(&self, _list: Vec<NodeInfo>) {}
}

struct MockRingClient {
    migrated: AtomicBool,
}

#[async_trait::async_trait]
impl RingClient for MockRingClient {
    async fn find_successor(&self, _addr: &str, _id: u64) -> Result<NodeInfo, Status> {
        Ok(NodeInfo::new(1, "node1"))
    }
    async fn get_predecessor(&self, _addr: &str) -> Result<Option<NodeInfo>, Status> {
        Ok(None)
    }
    async fn notify(&self, _addr: &str, _other: NodeInfo) -> Result<bool, Status> {
        Ok(true)
    }
    async fn get_successor_list(&self, _addr: &str) -> Result<Vec<NodeInfo>, Status> {
        Ok(vec![])
    }
    async fn ping(&self, _addr: &str) -> Result<bool, Status> {
        Ok(true)
    }
    async fn migrate(
        &self,
        _addr: &str,
        _sender: &str,
        _migrants: &[MigrantIndividual],
    ) -> Result<bool, Status> {
        self.migrated.store(true, Ordering::SeqCst);
        Ok(true)
    }
}

#[tokio::test]
async fn lead_migration_emigrates_on_interval() {
    let member = Arc::new(MockRingMember {
        self_node: NodeInfo::new(1, "self:50060"),
        successor: Mutex::new(Some(NodeInfo::new(2, "succ:50060"))),
    });
    let client = Arc::new(MockRingClient {
        migrated: AtomicBool::new(false),
    });
    let buffer = Arc::new(MigrantBuffer::new());
    let config = MigrationConfig {
        interval_generations: 5,
        migrant_count: 2,
    };
    let migration = LeadMigration::new(config, member, client.clone(), TopKSelector, buffer.clone());

    let pop = vec![vec![0.1], vec![0.2], vec![0.3]];
    let fit = vec![1.0, 3.0, 2.0];

    // Gen 4: no migration
    migration.maybe_emigrate(4, &pop, &fit).await.unwrap();
    assert!(!client.migrated.load(Ordering::SeqCst));

    // Gen 5: migration triggers
    migration.maybe_emigrate(5, &pop, &fit).await.unwrap();
    assert!(client.migrated.load(Ordering::SeqCst));

    // Test buffer drain
    buffer.push(vec![MigrantIndividual {
        genes: vec![0.9],
        fitness: 99.0,
    }]).await;
    let immigrants = migration.drain_immigrants().await;
    assert_eq!(immigrants.len(), 1);
    assert_eq!(immigrants[0].fitness, 99.0);
}

