# Use Case: Predict Key Locations with Learned Index (RMI)

## Description

Maps high-dimensional continuous data keys (such as Hilbert space-filling curve encodings) to 64-bit ring identifier positions ($[0, 2^{64}-1]$) using a Two-Stage Recursive Model Index (RMI), replacing uniform hashing with order-preserving data distribution mapping.

## Primary actor

- `LearnedIndex` / `RmiModel`

## Supporting systems

- `feature(key)` extraction
- Stage-0 partition bins
- Stage-1 leaf models (`LinearLeaf`, `RadixSplineLeaf`)
- `Anchor` compensation parameters (`scale`, `offset`)

## Main steps

1. **Feature Extraction**:
   - `feature(key)` extracts a normalized scalar $f \in [0.0, 1.0]$ from the key.
   - For Hilbert keys (e.g. `"0a3f8c...|{json}"`), it parses the hex prefix into a floating-point value in $[0, 1]$.
2. **Stage-0 Routing**:
   - The feature $f$ is mapped to a stage-0 bin index:
     $$\text{bin} = \min\left(\lfloor f \cdot \text{stage0\_bins} \rfloor, \text{stage0\_bins} - 1\right)$$
3. **Stage-1 Prediction**:
   - The selected leaf model $\text{leaves}[\text{bin}]$ predicts the cumulative distribution function (CDF) value $\hat{F}(f) \in [0.0, 1.0]$:
     - **Linear Leaf**: $\hat{F}(f) = \text{weight} \cdot f + \text{bias}$
     - **Radix Spline Leaf**: Spline interpolation between recorded knots.
4. **Anchor Correction**:
   - The leaf's `Anchor` applies online scale and offset adjustments:
     $$\text{cdf} = \text{scale} \cdot \hat{F}(f) + \text{offset}$$
   - The result is clamped to $[0.0, 1.0]$.
5. **Ring Space Scaling**:
   - The CDF value is scaled into the 64-bit identifier space:
     $$\text{NodeId} = \lfloor \text{cdf} \cdot 2^{64} \rfloor$$

## Postconditions

- Similar keys map to contiguous identifier regions on the ring.
- Range scans across the ring preserve topological proximity in multi-dimensional space.
