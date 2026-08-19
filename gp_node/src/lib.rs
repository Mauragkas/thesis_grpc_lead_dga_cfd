pub mod backend;
pub mod data;
pub mod domain;
pub mod model;

pub use backend::{BackendFactory, ComputeBackend, CpuOpenMpBackend, CudaBackend, GpDeviceType, RocmBackend};
pub use data::{DatasetLoader, DatasetSplitter, StandardScaler, TargetScaler};
pub use domain::{
    AircraftConfig, DatasetPartition, DatasetRecord, DatasetSplit, EvaluationMetrics,
    GpHyperparameters, KernelType, ACTIVE_FEATURE_NAMES,
};
pub use model::{GaussianProcessSurrogate, HyperparameterOptimizer, ReferenceKernel};
