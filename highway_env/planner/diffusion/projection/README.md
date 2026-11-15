# TRACE Joint Trajectory Projection (JTP)

This module implements the **Active-set Weighted Joint Jacobian Projection**
for control-Diffusion outputs. The projection is computed directly from the
Jacobian rows of the currently active constraint set. No SCP/QP constrained
optimization solver, OSQP, or SciPy sparse-matrix backend is used.

## Projection variables and constraints

The projection variable is the concatenated control sequence
`U = [a_x, delta_f]` for all jointly controlled vehicles. The frozen,
load-conditioned Koopman predictor maps each candidate control sequence to its
executed trajectories and chassis responses.

The JTP projection directly considers these constraint families:

- full-horizon chassis stability: yaw rate, sideslip angle, roll angle, LTR;
- road geometry: road-corridor boundary clearance and trajectory curvature;
- control magnitude: longitudinal acceleration and front-wheel steering angle.

External-actor and inter-vehicle collision avoidance are handled upstream by
Adaptive Anchor scoring/selection. Control-rate constraints are independently
evaluated in the paper evaluation but are excluded from the current lightweight
JTP update.

## Projection procedure

At each adaptive projection iteration, JTP evaluates the nonlinear Koopman
rollout, normalizes constraint residuals, and selects the most violated
constraints within each family. It computes Jacobian rows for this active set.
A violation-weighted, damped minimum-norm correction is obtained from the
regularized active-set Gram system. A per-control-node trust bound and
backtracking line search regulate the update. Each accepted control update is
re-evaluated through the original Koopman predictor and the retained nonlinear
constraints.

The local metric combines the trajectory-deviation curvature induced by the
Koopman execution Jacobian and the control regularizer. Damping stabilizes the
Gram system when active constraint gradients are correlated or nearly
dependent. Iteration is retained because the Koopman execution map and
geometric residuals are nonlinear; this does not introduce a QP solver.

## Output semantics

`JTPResult.controls`, `trajectories`, `states`, and `ltr` always contain the
best accepted JTP iterate and its Koopman rollout. The Boolean
`projector_constraints_satisfied` reports whether all constraint families
implemented inside this projector satisfy their configured tolerances. It is a
diagnostic field, not a full-platoon feasibility result. The independent
synchronized-frame evaluator additionally checks inter-truck/external
collisions and control rates.

A nominal control sequence that already satisfies the JTP constraint set is a
valid identity projection. If no candidate update improves the projection
merit, the best accepted iterate remains the nominal sequence. No external
fallback or replacement of a returned JTP result is performed.

## API

```python
from highway_env.planner.diffusion.projection import (
    JointTrajectoryProjector, RoadGeometry, load_jtp_config,
)

cfg = load_jtp_config("configs/projection/jtp.yaml")
projector = JointTrajectoryProjector(frozen_koopman_k2, cfg)
road = RoadGeometry(centerlines_xy=road_xy, left_width_m=6.0, right_width_m=6.0)
result = projector.project(nominal_controls, synchronized_k2_context, road)
```

`nominal_controls` has shape `[N, K, 2]` in SI units, with channels
`[acceleration (m/s^2), front-wheel steering (rad)]`. `RoadGeometry` represents
the drivable road corridor in world coordinates. The result contains projected
controls, their Koopman execution trajectories, chassis states, and diagnostics.

All constraint thresholds, normalization scales, active-set sizes, damping,
trust-region, line-search, and correction-reporting parameters are centralized
in `configs/projection/jtp.yaml`.

## Dependencies

The JTP projection module requires PyTorch and PyYAML.
