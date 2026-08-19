# Recursive Model Index (RMI) Module

Two-stage learned index architecture mapping high-dimensional key features to 64-bit ring hash positions.

## Files and Code Structure

### 1. `model.rs`
- **`PidState`** (struct):
  - `pub proportional: u8`: Proportional error state ($0..=3$).
  - `pub integral: u8`: Quantized integral accumulator ($0..=15$).
  - `pub derivative_sign: bool`: Sign of derivative contribution.
  - `pub derivative_mag: u8`: Magnitude of derivative contribution ($0..=1$).
- **`RmiModel`** (struct, serde Serialize/Deserialize):
  - `pub stage0_bins: usize`: Number of partition bins (default: `16`).
  - `pub leaves: Vec<LeafKind>`: Stage-1 leaf models.
  - `pub n: usize`: Number of training keys.
  - `pub version: u64`: Model version number.
  - `pub pid_state: Vec<PidState>`: Per-leaf PID controller state.
  - `pub fn predict(&self, key: &str) -> NodeId`:
    1. Extracts normalized feature $f = \text{feature}(key)$.
    2. Routes to bin: $\text{bin} = \min(\lfloor f \cdot \text{stage0\_bins} \rfloor, \text{stage0\_bins}-1)$.
    3. Evaluates leaf: $\text{cdf} = \text{leaves}[\text{bin}].\text{predict}(f)$.
    4. Scales to 64-bit ring: $\text{NodeId} = \lfloor \text{cdf} \cdot 2^{64} \rfloor$.

---

### 2. `feature.rs`
- **`pub fn feature(key: &str) -> f64`**:
  - Parses the hex prefix of `key` (before the pipe `|` delimiter) and converts it into a floating-point scalar normalized to $[0.0, 1.0]$.

---

### 3. `leaf.rs`
- **`Anchor`** (struct):
  - `pub offset: f64`: Additive prediction offset.
  - `pub scale: f64`: Multiplicative prediction scale factor.
  - `pub fn apply(&self, pred: f64) -> f64`: Computes $\text{clamp}(pred \cdot \text{scale} + \text{offset}, 0.0, 1.0)$.
- **`LeafKind`** (enum):
  - `Linear(LinearLeaf)`
  - `RadixSpline(RadixSplineLeaf)`
- **`LinearLeaf`** (struct):
  - `pub weight: f64`, `pub bias: f64`, `pub anchor: Anchor`.
  - `pub fn predict(&self, f: f64) -> f64`: Evaluates $\text{anchor.apply}(\text{weight} \cdot f + \text{bias})$.
- **`RadixSplineLeaf`** (struct):
  - `pub points: Vec<(f64, f64)>`, `pub radix_table: Vec<usize>`, `pub anchor: Anchor`.
  - `pub fn predict(&self, f: f64) -> f64`: Fast $O(1)$ radix table lookup followed by spline interpolation.

---

### 4. `train.rs`
- **`RmiModel::train_auto(keys: &[String], version: u64) -> RmiModel`**:
  - Automatically selects optimal bin count ($\max(8, \min(64, |keys| / 10))$) and fits linear or spline models against sorted empirical CDF ranks.

---

### 5. `diff.rs`
- **`LeafDiff`** (struct):
  - `pub index: usize`: Leaf bin index.
  - `pub weight: f64`, `pub bias: f64`, `pub anchor_offset: f64`.
  - Used for differential model synchronization during federated training rounds.
