pub mod common;
pub mod gp;
pub mod knn;
pub mod mlp;
pub mod rf;

// Common exports
pub use common::{
    get_best_available_device, probe_available_devices, BackendError, GpDeviceInfo,
    GpDeviceInfoFFI, GpDeviceType, GpStatusCode,
};

// GP backend exports
pub use gp::{
    BackendFactory, ComputeBackend, CpuOpenMpBackend, CudaBackend, GpHyperparamsFFI,
    GpKernelType, RocmBackend,
};

// k-NN backend exports
pub use knn::{
    create_knn_cpu, create_knn_cuda, create_knn_rocm, KnnBackend, KnnBackendFactory,
    KnnCpuBackend, KnnCudaBackend, KnnModelHandle, KnnRocmBackend, KnnStatusCode,
};

// RF backend exports
pub use rf::{
    RfBackend, RfBackendFactory, RfCpuBackend, RfCudaBackend, RfHyperparamsFFI, RfModelHandle,
    RfRocmBackend, RfStatusCode,
};

// MLP (Neural Network) backend exports
pub use mlp::{
    MlpActivationFFI, MlpBackend, MlpBackendFactory, MlpCpuBackend, MlpCudaBackend,
    MlpHyperparamsFFI, MlpModelHandle, MlpRocmBackend, MlpStatusCode,
};

