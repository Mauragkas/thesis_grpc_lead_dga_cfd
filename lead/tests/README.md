# LEAD Unit and Integration Tests

Unit and integration tests for the learned indexing and PID tuning modules of the `lead-node` crate.

## Test Files and Test Cases

### 1. `learned_index.rs`
- `next_version_starts_at_one`: Verifies that freshly initialized models start at version `1`.
- `in_grace_just_after_construction`: Tests that startup grace period suppresses spurious early drift triggers.
- `pending_and_active_visibility_through_current_model`: Tests candidate model staging in `pending` before `activate()`.
- `record_insert_marks_dirty_leaf`: Tests that key insertions correctly flag the corresponding leaf index in `dirty_leaves` for differential sync.
- `accept_pushed_model_rejects_rollback_bad_payload_then_accepts`: Tests validation logic preventing model version downgrades or corrupted payloads.
- `record_insert_raises_update_ready_after_threshold`: Validates that exceeding 40% drift with $\ge 50$ keys raises `update_ready = true`.
- `reset_drift_clears_update_ready`: Validates that `reset_drift()` restores clean state.
- `record_insert_returns_pid_due_at_interval`: Tests that `record_insert` returns `true` every `PID_ADJUST_INTERVAL` (100) insertions.

---

### 2. `pid_tuner.rs`
- `all_outside_scales_down_and_resets_integral`: Validates that persistent inaccurate predictions trigger downward scale adjustments ($\Delta \text{scale} = -0.05$) and reset the integral accumulator.
- `below_min_samples_is_noop`: Confirms no PID adjustment occurs when sample count $< 20$.
- `default_constants_match_original_magic_numbers`: Asserts default tuning parameters (`target_ratio=0.95`, `scale_step=0.05`, `centering_step=0.01`).
- `on_target_accumulates_integral_without_scaling`: Tests integral error accumulation when predictions remain on target.
