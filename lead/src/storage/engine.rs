use std::path::Path;

use async_trait::async_trait;

use super::{InMemoryStore, KeyStore, SledStore};

/// Unified storage engine enum capable of runtime switching between
/// in-memory and disk-backed (sled) storage (Strategy pattern).
pub enum StorageEngine {
    Memory(InMemoryStore),
    Sled(SledStore),
}

impl StorageEngine {
    pub fn memory() -> Self {
        Self::Memory(InMemoryStore::new())
    }

    pub fn sled<P: AsRef<Path>>(path: P) -> Result<Self, sled::Error> {
        Ok(Self::Sled(SledStore::open(path)?))
    }

    pub fn sled_temporary() -> Result<Self, sled::Error> {
        Ok(Self::Sled(SledStore::open_temporary()?))
    }
}

#[async_trait]
impl KeyStore for StorageEngine {
    async fn get(&self, key: &str) -> Option<String> {
        match self {
            Self::Memory(s) => s.get(key).await,
            Self::Sled(s) => s.get(key).await,
        }
    }

    async fn put(&self, key: String, val: String) {
        match self {
            Self::Memory(s) => s.put(key, val).await,
            Self::Sled(s) => s.put(key, val).await,
        }
    }

    async fn remove(&self, key: &str) -> Option<String> {
        match self {
            Self::Memory(s) => s.remove(key).await,
            Self::Sled(s) => s.remove(key).await,
        }
    }

    async fn range_scan(&self, start_key: &str, count: usize) -> Vec<(String, String)> {
        match self {
            Self::Memory(s) => s.range_scan(start_key, count).await,
            Self::Sled(s) => s.range_scan(start_key, count).await,
        }
    }

    async fn range_scan_after(&self, after_key: &str, count: usize) -> Vec<(String, String)> {
        match self {
            Self::Memory(s) => s.range_scan_after(after_key, count).await,
            Self::Sled(s) => s.range_scan_after(after_key, count).await,
        }
    }

    async fn snapshot(&self) -> Vec<(String, String)> {
        match self {
            Self::Memory(s) => s.snapshot().await,
            Self::Sled(s) => s.snapshot().await,
        }
    }

    async fn len(&self) -> usize {
        match self {
            Self::Memory(s) => s.len().await,
            Self::Sled(s) => s.len().await,
        }
    }

    async fn is_empty(&self) -> bool {
        match self {
            Self::Memory(s) => s.is_empty().await,
            Self::Sled(s) => s.is_empty().await,
        }
    }

    async fn flush(&self) {
        match self {
            Self::Memory(s) => s.flush().await,
            Self::Sled(s) => s.flush().await,
        }
    }
}
