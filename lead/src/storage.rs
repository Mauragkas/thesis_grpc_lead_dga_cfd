use std::collections::BTreeMap;
use std::ops::Bound;

use async_trait::async_trait;
use tokio::sync::RwLock;

/// Ordered storage interface (Step 3). Backed by BTreeMap so contiguous range
/// scans are O(k) and key order approximates LearnedHASH order.
#[async_trait]
pub trait KeyStore: Send + Sync {
    async fn get(&self, key: &str) -> Option<String>;
    async fn put(&self, key: String, val: String);
    async fn remove(&self, key: &str) -> Option<String>;
    async fn range_scan(&self, start_key: &str, count: usize) -> Vec<(String, String)>;
    async fn range_scan_after(&self, after_key: &str, count: usize) -> Vec<(String, String)>;
    async fn snapshot(&self) -> Vec<(String, String)>;
    async fn len(&self) -> usize;
    async fn is_empty(&self) -> bool;
}

pub struct InMemoryStore {
    data: RwLock<BTreeMap<String, String>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(BTreeMap::new()),
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

    async fn range_scan(&self, start_key: &str, count: usize) -> Vec<(String, String)> {
        self.data
            .read()
            .await
            .range(start_key.to_string()..)
            .take(count)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    async fn range_scan_after(&self, after_key: &str, count: usize) -> Vec<(String, String)> {
        self.data
            .read()
            .await
            .range((Bound::Excluded(after_key.to_string()), Bound::Unbounded))
            .take(count)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    async fn snapshot(&self) -> Vec<(String, String)> {
        self.data
            .read()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    async fn len(&self) -> usize {
        self.data.read().await.len()
    }

    async fn is_empty(&self) -> bool {
        self.data.read().await.is_empty()
    }
}
