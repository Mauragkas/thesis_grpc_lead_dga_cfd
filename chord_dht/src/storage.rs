use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::ring::{hash, in_range, NodeId};

/// Interface segregation: only the operations the ring + handlers need.
/// Dependency inversion: `ChordNode` depends on this, not on `HashMap`.
#[async_trait]
pub trait KeyStore: Send + Sync {
    async fn get(&self, key: &str) -> Option<String>;
    async fn put(&self, key: String, val: String);
    async fn remove(&self, key: &str) -> Option<String>;
    async fn keys_in_range(
        &self,
        start: NodeId,
        end: NodeId,
        inclusive_end: bool,
    ) -> Vec<(String, String)>;
    async fn snapshot(&self) -> HashMap<String, String>;
}

/// Open-closed: a different backend (disk, DB) can be added without touching
/// `ChordNode` or the handlers.
pub struct InMemoryStore {
    data: RwLock<HashMap<String, String>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl KeyStore for InMemoryStore {
    async fn get(&self, key: &str) -> Option<String> {
        self.data.read().await.get(key).cloned()
    }

    async fn put(&self, key: String, val: String) {
        self.data.write().await.insert(key, val);
    }

    async fn remove(&self, key: &str) -> Option<String> {
        self.data.write().await.remove(key)
    }

    async fn keys_in_range(
        &self,
        start: NodeId,
        end: NodeId,
        inclusive_end: bool,
    ) -> Vec<(String, String)> {
        self.data
            .read()
            .await
            .iter()
            .filter(|(k, _)| in_range(hash(k), start, end, inclusive_end))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    async fn snapshot(&self) -> HashMap<String, String> {
        self.data.read().await.clone()
    }
}
