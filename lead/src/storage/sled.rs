use std::ops::Bound;
use std::path::Path;

use async_trait::async_trait;

use super::KeyStore;

/// Durable, disk-backed key-value store powered by sled.
pub struct SledStore {
    db: sled::Db,
}

impl SledStore {
    /// Opens or creates a sled database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, sled::Error> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let db = sled::open(p)?;
        Ok(Self { db })
    }

    /// Creates a temporary sled database (useful for testing or ephemeral runs).
    pub fn open_temporary() -> Result<Self, sled::Error> {
        let config = sled::Config::new().temporary(true);
        let db = config.open()?;
        Ok(Self { db })
    }
}

#[async_trait]
impl KeyStore for SledStore {
    async fn get(&self, key: &str) -> Option<String> {
        match self.db.get(key.as_bytes()) {
            Ok(Some(val)) => String::from_utf8(val.to_vec()).ok(),
            _ => None,
        }
    }

    async fn put(&self, key: String, val: String) {
        if let Err(e) = self.db.insert(key.as_bytes(), val.as_bytes()) {
            tracing::error!("sled put error for key {key}: {e}");
        }
    }

    async fn remove(&self, key: &str) -> Option<String> {
        match self.db.remove(key.as_bytes()) {
            Ok(Some(val)) => String::from_utf8(val.to_vec()).ok(),
            _ => None,
        }
    }

    async fn range_scan(&self, start_key: &str, count: usize) -> Vec<(String, String)> {
        self.db
            .range(start_key.as_bytes()..)
            .take(count)
            .filter_map(|res| {
                let (k, v) = res.ok()?;
                let key = String::from_utf8(k.to_vec()).ok()?;
                let val = String::from_utf8(v.to_vec()).ok()?;
                Some((key, val))
            })
            .collect()
    }

    async fn range_scan_after(&self, after_key: &str, count: usize) -> Vec<(String, String)> {
        self.db
            .range::<&[u8], _>((Bound::Excluded(after_key.as_bytes()), Bound::Unbounded))
            .take(count)
            .filter_map(|res| {
                let (k, v) = res.ok()?;
                let key = String::from_utf8(k.to_vec()).ok()?;
                let val = String::from_utf8(v.to_vec()).ok()?;
                Some((key, val))
            })
            .collect()
    }

    async fn snapshot(&self) -> Vec<(String, String)> {
        self.db
            .iter()
            .filter_map(|res| {
                let (k, v) = res.ok()?;
                let key = String::from_utf8(k.to_vec()).ok()?;
                let val = String::from_utf8(v.to_vec()).ok()?;
                Some((key, val))
            })
            .collect()
    }

    async fn len(&self) -> usize {
        self.db.len()
    }

    async fn is_empty(&self) -> bool {
        self.db.is_empty()
    }

    async fn flush(&self) {
        if let Err(e) = self.db.flush() {
            tracing::error!("sled flush error: {e}");
        }
    }
}
