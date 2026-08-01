use crate::gene_store::record::GeneRecord;

/// SRP / OCP: decides *which* records to evict. New policies
/// (frequency-based, size-cap, hybrid) are added by implementing this
/// trait without touching `InMemoryGeneStore`.
pub trait EvictionPolicy: Send + Sync {
    fn should_evict(&self, record: &GeneRecord, current_generation: usize) -> bool;
}

/// Access-refresh TTL: a record is evicted only if it has not been
/// retrieved for more than `max_age` generations. Default `max_age = 5`.
#[derive(Debug, Clone, Copy)]
pub struct GenerationEvictor {
    pub max_age: usize,
}

impl Default for GenerationEvictor {
    fn default() -> Self {
        Self { max_age: 5 }
    }
}

impl EvictionPolicy for GenerationEvictor {
    fn should_evict(&self, record: &GeneRecord, current_generation: usize) -> bool {
        current_generation.saturating_sub(record.last_accessed_gen) > self.max_age
    }
}
