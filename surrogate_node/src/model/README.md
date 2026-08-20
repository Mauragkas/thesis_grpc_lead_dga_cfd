# Surrogate Models Module

Contains domain implementations of high-level aerodynamic surrogate models.

## Available Models

### 1. `mlp/` — `MlpSurrogate`
- **Architecture**: Multi-Layer Perceptron neural network with configurable hidden layer topologies (default `[64, 32]`).
- **Activation Functions**: SiLU, ReLU, GELU, Tanh.
- **Optimization**: Adam optimizer with weight decay regularization and mini-batching.
- **Hardware**: Native OpenMP CPU multi-threading, CUDA kernels, and ROCm stubs.

---

### 2. `gp/` — `GaussianProcessSurrogate`
- **Architecture**: Gaussian Process regression with Matérn 5/2, RBF (Squared Exponential), or Exponential kernels.
- **Optimization**: Maximum Marginal Likelihood (MML) hyperparameter tuning with length scales per dimension.
- **Inference**: Analytical mean prediction and posterior uncertainty standard deviation $\sigma(\mathbf{x}^*)$.

---

### 3. `rf/` — `RfSurrogate`
- **Architecture**: Random Forest ensemble regressor.
- **Inference**: Mean target prediction and inter-tree variance estimation for epistemic uncertainty.

---

### 4. `knn/` — `KnnSurrogate`
- **Architecture**: Exact Euclidean k-Nearest Neighbors regressor with distance-weighted interpolation.
