# TRACE Joint Trajectory Projection (JTP)

JTP keeps the batch `[N,40,2]` runtime interface and projects each vehicle independently. Every single-vehicle optimization contains 80 control variables; no 240-dimensional platoon QP/LS problem is constructed. Inter-truck and external-actor collision constraints are excluded from projection and remain in the independent evaluation module.

## Configuration

`configs/projection/jtp.yaml` selects the solver and sparse temporal sampling:

```yaml
projection:
  solver_method: constrained_qp  # damped_least_squares | constrained_qp
  qp_group_order: [road_curvature, stability, control_rate]
  constraint_sampling_stride: 2  # 1, 2, or 4
```

Stride 1/2/4 initially samples 40/20/10 G1/G2 time points. The terminal point is always included. G3 control-rate constraints and acceleration/steering hard bounds retain the full 40-step horizon. Stage 2 always checks every nonlinear constraint row over the full horizon; violations at unsampled G1/G2 times are inserted into subsequent solver iterations.

## Solver backends

- `damped_least_squares`: retains the existing weighted damped least-squares backend, active-set/Top-K row selection, bounded Cholesky active-set solve, Trust Region, metric half-space protection, and Backtracking.
- `constrained_qp`: retains the OSQP Full-Group QP backend. Every row in the current sparse temporal pool is included for its priority group; there is no Top-K truncation or Trust Region. State slack, common tolerance handling, cross-group protection, and hard control bounds remain enabled.

For `constrained_qp`, `projection.qp_group_order` selects the order of `stability`, `road_curvature`, and `control_rate`; all six permutations are supported. The default is `[road_curvature, stability, control_rate]`. Group labels G1/G2/G3 follow the configured list positions. The DLS backend always retains its original order: stability, road and curvature, then control continuity. The K2 execution model, dynamic road association, Backtracking, and complete Stage-2 nonlinear acceptance are shared.

## API

`JointTrajectoryProjector(execution_model, config).project(nominal_controls, context, road_geometry) -> JTPResult` retains the batch input/output contract. The projector internally slices each vehicle's controls, K2 context, and road geometry, solves each 80-dimensional problem independently, and concatenates the results.

`evaluate_preprojection(nominal_controls, context, road_geometry)` evaluates the full-horizon projector constraint families without modifying controls.

## Evaluation

Collision metrics are computed independently by `evaluation/ablation/jtp/metrics.py`; they are not passed to the projector. Install the optional QP dependencies with `pip install -e ".[jtp]"` when `solver_method: constrained_qp` is selected.
