# TRACE Joint Trajectory Projection (JTP)

This module implements a **Active-set weighted joint projection**
for TRACE control-diffusion outputs. It does not use SCP, QP assembly, OSQP,
or SciPy sparse matrices.

## Projection variables and constraints

The projection variable is the concatenated control sequence
`U = [a_x, delta_f]` for all jointly controlled vehicles. The frozen,
load-conditioned Koopman predictor maps each candidate control sequence to its
executed trajectory and chassis responses.

The active constraint families are:

- full-horizon chassis stability: yaw rate, sideslip angle, roll angle, LTR;
- road geometry: road-corridor boundary clearance and trajectory curvature;
- control magnitude: longitudinal acceleration and front-wheel steering angle.

External-actor and inter-vehicle collision avoidance are intentionally handled
upstream by Adaptive Anchor scoring/selection. Control-rate constraints are
excluded from the current lightweight JTP profile.

## Algorithm

At each adaptive iteration, JTP evaluates the nonlinear Koopman rollout,
normalizes constraint residuals, and selects the most violated constraints
within each family. It computes Jacobian rows only for this active set. A
violation-weighted, damped minimum-norm correction is then obtained from a
small regularized Gram system. A per-control-node trust bound and backtracking
line search regulate the update. The updated controls are rolled through the
original Koopman predictor and all retained nonlinear constraints are
re-evaluated before acceptance.

The local metric combines the trajectory-deviation curvature induced by the
Koopman execution Jacobian and the control regularizer. The damping term keeps
the active-set Gram system numerically stable when constraint gradients are
correlated or nearly dependent.

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
trust-region and line-search parameters are centralized in
`configs/projection/jtp.yaml`. Values are provisional engineering defaults and
must be calibrated after the algorithm is frozen.

## Dependencies

The JTP projection module requires PyTorch and PyYAML. **OSQP and SciPy are no
longer required.**
