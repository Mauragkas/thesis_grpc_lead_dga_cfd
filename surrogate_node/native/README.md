# Native C++/CUDA Engine

High-performance native implementations of mathematical operations, Cholesky decomposition, decision trees, and neural network layers.

## Structure

- **`incl/`**: C++ header declarations:
  - `common/`: Common device abstractions (`gp_device.h`, `gp_types.h`).
  - `mlp/`: `mlp_backend.h` (forward/backward passes, layer activations, Adam optimizer).
  - `gp/`: `gp_backend.h`, `gp_cpu.h`, `gp_cuda.h`, `gp_rocm.h`.
  - `rf/`: `rf_backend.h`, `rf_internal.h`.
  - `knn/`: `knn_backend.h`, `knn_internal.h`.
- **`src/`**: Implementation files:
  - `mlp/`: `mlp_cpu.cpp` (OpenMP-accelerated matrix multiplication and activations), `mlp_cuda.cu` (CUDA kernels).
  - `gp/`: `gp_cpu.cpp` (OpenMP covariance and Cholesky solver), `gp_cuda.cu`.
  - `rf/`: `rf_cpu.cpp` (multi-threaded tree building), `rf_cuda.cu`.
  - `knn/`: `knn_cpu.cpp`, `knn_cuda.cu`.

## Build Integration

Compiled at build time by `surrogate_node/build.rs` using `cc`:
- Checks for `nvcc` or `/opt/cuda/bin/nvcc`. If present, compiles `.cu` files into `gp_cuda_native` and links `cudart`.
- Otherwise, compiles CPU OpenMP sources (`gp_native`) with `-fopenmp` and links `libgomp`.
