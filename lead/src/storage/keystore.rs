use async_trait::async_trait;

/// Ordered storage interface (Step 3). Backed by BTreeMap or sled so contiguous range
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
    async fn flush(&self) {}

    /// Read internal node metadata (isolated from user keyspace).
    async fn get_meta(&self, _key: &str) -> Option<String> {
        None
    }

    /// Persist internal node metadata (isolated from user keyspace).
    async fn put_meta(&self, _key: String, _val: String) {}
}
