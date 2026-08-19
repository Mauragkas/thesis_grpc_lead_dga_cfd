# Use Case: Tune Leaf Models with Online PID Controller

## Description

Performs lightweight, online tuning of individual leaf model anchors (`scale` and `offset`) using a 2-bit PID controller. Adjusts model predictions in real time based on observed lookup error rates, adapting to local workload shifts between full retraining cycles.

## Primary actor

- `PidTuner`

## Supporting systems

- `PidState` (proportional, integral, and derivative state)
- `Anchor` (`scale`, `offset`)
- `LearnedIndex`

## Preconditions

- Key insertions and queries record accuracy statistics for active leaves.

## Main steps

1. Key insertions call `record_insertion(key)` on `LeadNode`.
2. `LearnedIndex` marks the corresponding leaf bin dirty and updates sample counters.
3. Every `PID_ADJUST_INTERVAL` (100) insertions, `LeadNode` triggers `run_pid_adjustment()`.
4. For each leaf with sufficient observations ($\ge 20$ samples):
   1. Compute error: $e = \text{observed\_accuracy} - \text{target\_accuracy}$ (target: $0.95$).
   2. **Proportional Term**: Classify error magnitude into discrete adjustment bands.
   3. **Integral Term**: Accumulate persistent error offsets.
   4. **Derivative Term**: Track rate of error change.
   5. **Parameter Adjustment**:
      - If under-predicting / too narrow: Adjust `anchor.scale` by `+scale_step` ($0.05$).
      - If over-predicting / too wide: Adjust `anchor.scale` by `-scale_step` ($0.05$).
      - Adjust `anchor.offset` by `centering_step` ($0.01$) to center the prediction window.
      - Clamp `anchor.scale` within safe bounds $[0.5, 2.0]$.

## Postconditions

- Leaf prediction windows automatically widen or narrow to maintain 95% target routing accuracy.
- Local distribution drift is mitigated dynamically without invoking expensive model retraining.
