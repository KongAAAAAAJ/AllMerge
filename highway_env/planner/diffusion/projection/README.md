# TRACE Joint Trajectory Projection (Stage 1)

The only runtime projection algorithm is priority-grouped sequential JTP with
cross-group non-degradation and outer nonlinear validation. All seven constraint
families remain enabled: stability (yaw, sideslip, roll, LTR), road, curvature,
control magnitude, inter-truck collision, external-actor collision, control rate.

## Solver configuration

`configs/projection/jtp.yaml`, `projection.solver_method: damped_least_squares`
is the implemented Stage-1 backend. `constrained_qp` is a reserved Stage-2
backend: selecting it raises an explicit NotImplementedError. No silent fallback
or legacy ungrouped solver is provided. `group_count` is 2 or 3. Initial family
priority is `q * (1 + max(relu(normalized_residual)))`; weights are provisional,
neutral by default, configurable per family. Sort descending and split into
near-equal groups, fixed for one project() call. Ties use existing family order.

## Execution

- `linearization.py`: cache conditional A/B, z0, interpolation, and analytic
  latent sensitivities across the call. Use six batched pointwise decoder VJPs
  and trapezoidal pose sensitivity. Recompute output/constraint derivatives at
  the outer working point; freeze road association during linearization only.
  The existing callable adapter contract uses one shared autograd output Jacobian.
- `grouped_solver.py`: reuse one objective-metric Cholesky factorization, then
  solve each group's active residuals by the original weighted damped Gram
  Cholesky method (lstsq only if factorization fails). After each group, cyclic
  metric halfspace projections enforce previous selected rows J_H d <= 0
  within tolerance. If protection cannot be verified, discard that increment.
  Scale the incremental direction to fit cumulative normalized trust bounds;
  recheck protection after scaling. No extra K2 calls inside the groups.
- `joint_projector.py`: full nonlinear validation with refreshed geometry and
  all constraints after each backtracking trial; accept only when all priority
  group maximum excesses do not increase within tolerance and one improves,
  or when all internal constraints satisfy their own tolerances. Return last
  accepted controls, even if infeasible; no external fallback. Initial out-of-box
  controls may trigger the existing box guard; the guarded candidate is checked
  by the same nonlinear rule. In this exceptional case the guard can exceed the
  local trust step. Nominal-feasible inputs return unchanged.

The LS backend is approximate and does not exactly minimize total trajectory
deviation under inequalities. Protection covers selected previous rows in the
current linearized model; nonlinear acceptance is on group maxima (not every
individual residual). Active-set caps, trust budget and priority conflicts can
prevent progress. Neither LS nor local nonlinear checks guarantee feasibility
of every input or physical TruckSim safety. Independent paper metrics are unchanged.

## API

`JointTrajectoryProjector(execution_model, config).project(nominal_controls,
context, road_geometry, external_actors=()) -> JTPResult` and
`evaluate_preprojection(...)` retain their existing signatures. No checkpoint,
K2 network, training or evaluation metric modification is required.

## Tests

`python -m pytest -q tests/test_joint_trajectory_projection.py tests/test_jtp_grouped_stage1.py`
