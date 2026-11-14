# Joint Trajectory Projection (JTP)

This directory contains the independent SCP/OSQP implementation of the final
TRACE **Joint Trajectory Projection** formulation. The previous curvature-only port has been removed from the active source tree.

## Mathematical contract

For the jointly planned trucks, JTP optimizes executed trajectories `tau_i`
and control sequences `U_i`:

- objective: weighted deviation from nominal Koopman execution trajectories
  plus a control-sequence regularizer;
- dynamics consistency: `tau_i = T_i(U_i)`;
- full-horizon chassis stability: yaw rate, sideslip angle, roll angle, LTR;
- road geometry: footprint-circle road-boundary clearance and trajectory
  curvature;
- collision avoidance: pairwise controlled-truck footprints and predicted
  surrounding-traffic footprints;
- actuator constraints: acceleration, steering, and their rates of change.

Each SCP subproblem has both `Delta_tau` and `Delta_U` as decision variables.
All linearized constraints are assembled in one sparse QP. After a control
update, the frozen Koopman predictor is called again and the original nonlinear
constraints are checked.

## Current project interface

```python
from highway_env.planner.diffusion.projection import (
    JointTrajectoryProjector,
    RoadGeometry,
    ExternalTraffic,
    constant_velocity_poses,
    load_jtp_config,
)
from highway_env.planner.diffusion.common.execution import VehicleExecution

cfg = load_jtp_config("configs/projection/jtp.yaml")
execution = VehicleExecution(
    {"model": "koopman_k2", "checkpoint": "outputs/koopman/checkpoints/v0/k2/best.pt"},
    dt=cfg["execution"]["dt_s"],
    wheelbase_m=cfg["execution"]["wheelbase_m"],
)
projector = JointTrajectoryProjector(execution, cfg)

# nominal_controls: [N,40,2], SI units [ax (m/s^2), front-wheel delta (rad)]
# context: five batched K2 context fields, each with leading dimension N
# road: world-frame centerline polyline and left/right drivable half-widths
road = RoadGeometry(centerlines_xy=road_xy, left_width_m=6.0, right_width_m=6.0)

# Optional surrounding traffic: initial states [A,5] = [x,y,vx,vy,yaw]
external_poses = constant_velocity_poses(
    external_initial_states, horizon_steps=40, dt_s=0.1
)
external = ExternalTraffic(external_poses)
result = projector.project(nominal_controls, context, road, external)
```

`RoadGeometry` accepts one or more world-frame road-corridor centerlines. Each
centerline must represent a connected drivable carriageway region bounded by
the outer road edges. For a multi-lane carriageway, use one centerline spanning
the full drivable width, rather than treating each lane centerline as an
independent corridor. Separate carriageways can be supplied as `[L,P,2]`
centerlines. Polyline point order defines the positive lateral direction. A
symmetric road corridor uses equal left/right half-widths.

The current TRACE model uses 40 control nodes and 40 execution steps at 10 Hz,
so the default control interpolation matrix is identity. If a future planner
uses `K != H`, provide the fixed interpolation matrix `C[H,K]` to the
`JointTrajectoryProjector` constructor.

## Provisional parameter status

All thresholds, empirical coefficients, weights, trust-region settings, and
QP tolerances are centralized in `configs/projection/jtp.yaml`. They are
engineering starting points only and **must be calibrated against the final
paper Experimental Setup before final experiments**. The YAML is the single
source of truth; do not duplicate constraint thresholds in Python code.

## Dependencies

Install manually in the project environment:

```bash
python -m pip install "osqp>=1.0.0" "scipy>=1.10.0" "PyYAML>=6.0"
```

Or install the declared project extra:

```bash
python -m pip install -e ".[jtp]"
```

OSQP is imported lazily, so configuration, geometry, constraint evaluation,
and QP assembly can be tested without it. A nontrivial SCP solve requires
OSQP.

## Important scope boundary

This is **Stage 1: an independent solver**. It does not change the diffusion
runtime, anchor scoring, training loop, or online feature builder. Runtime
history buffering, road-map conversion, and planner integration belong to
Stage 2.
