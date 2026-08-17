pub mod eviction;
pub mod in_memory;
pub mod metric;
pub mod record;
pub mod r#trait;

pub use eviction::{EvictionPolicy, GenerationEvictor};
pub use in_memory::InMemoryGeneStore;
pub use metric::{DistanceMetric, EuclideanDistance};
pub use r#trait::GeneStore;
pub use record::GeneRecord;
