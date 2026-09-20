# Stage 3 — Native 10 Hz Expert Dense Trajectory

Marker: `STAGE3_DENSE_EXPERT_V2`

This stage stores the Polynomial planner's native 10 Hz path before sparse 0.5 s reduction.

Contract:

- `future_trajectory_dense`: float32 `[40,2]`, timestamps 0.1..4.0 s
- `dense_dt`: float32 scalar 0.1
- `trajectory_horizon_s`: float32 scalar 4.0
- sparse `expert_trajectory_xy`: `[8,2]`, sampled directly from dense indices `[4,9,...,39]`
- no sparse-to-dense interpolation fallback

The old dataset remains readable. `validate_expert_dataset.py --require-dense` enforces the v2 dense schema.
