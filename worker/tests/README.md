# Worker Tests

Unit tests for the Python aerodynamic evaluation pipeline.

## Test Files and Test Cases

### 1. `conftest.py`
- Fixture `sample_genes()`: Provides a valid normalized 10-element gene vector within $[0, 1]$.
- Fixture `baseline_params()`: Provides reference dimensional parameters.

### 2. `test_aero.py`
- `test_solve_trim_interpolates_correctly()`: Tests linear interpolation of trimmed $\alpha$ and $(L/D)$ when $C_{L,\text{req}}$ lies within the sweep range.
- `test_solve_trim_returns_none_when_out_of_bounds()`: Tests that untrimmable flight conditions return `None`.
- `test_aero_evaluator_produces_valid_result()`: Tests end-to-end `AerosandboxAeroEvaluator.evaluate()` on standard aircraft parameters.

### 3. `test_geometry.py`
- `test_decode_genes_respects_bounds()`: Ensures every decoded parameter strictly lies within its `[min_val, max_val]` bound.
- `test_fuselage_volume_monotonicity()`: Ensures fuselage volume increases with fuselage length and radius parameters.
- `test_calculate_total_mass_and_weight()`: Validates material density and gravity scaling.

### 4. `test_fitness.py`
- `test_fitness_rejects_insufficient_volume()`: Confirms `REJECT_FITNESS` (`-1e9`) is returned when fuselage volume $< V_{\text{min}}$.
- `test_fitness_rejects_untrimmable_aero()`: Confirms rejection when aero solver returns `None`.
- `test_fitness_penalties_decrease_score()`: Validates that non-zero trim angles $\alpha_{\text{trim}}$ and positive $C_{m_\alpha}$ reduce fitness score according to configured weights.
