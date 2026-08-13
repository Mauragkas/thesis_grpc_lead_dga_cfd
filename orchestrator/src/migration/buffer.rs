//! SRP: a thread-safe buffer for incoming migrants. The gRPC `Migrate`
//! handler pushes here; the GA loop drains each generation.

use crate::migration::MigrantIndividual;
use tokio::sync::Mutex;

#[derive(Debug, Default)]
pub struct MigrantBuffer {
    inner: Mutex<Vec<MigrantIndividual>>,
}

impl MigrantBuffer {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Vec::new()),
        }
    }

    pub async fn push(&self, migrants: Vec<MigrantIndividual>) {
        let mut buf = self.inner.lock().await;
        buf.extend(migrants);
    }

    pub async fn drain(&self) -> Vec<MigrantIndividual> {
        let mut buf = self.inner.lock().await;
        std::mem::take(&mut *buf)
    }

    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }
}
