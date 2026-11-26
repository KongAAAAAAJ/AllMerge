# TRACE Joint Trajectory Projection (JTP)

JTP supports two priority-grouped solver backends selected by
`configs/projection/jtp.yaml`:

- `damped_least_squares`: the existing Stage-3 Cholesky grouped-LS backend,
  including its fixed control box, cumulative Trust Region, and metric
  half-space protection.
- `constrained_qp`: three sequential OSQP subproblems with a control-correction
  objective, hard control bounds, bounded penalized state slacks, and explicit
  bounded penalized protection slacks. The QP backend has no Trust Region.

When the nominal controls violate the magnitude box, QP first solves a separate
control-only restoration QP. It then solves the three priority groups.

Both backends share the six state-constraint families, priority ranking,
active-row selection, K2 Jacobian construction, road-association freezing,
Backtracking, and Stage-2 full-resolution nonlinear acceptance. Selecting QP
requires the optional dependencies installed with `pip install -e ".[jtp]"`.
There is no automatic solver fallback.

## QP formulation

For priority group `k`, the incremental control correction is `d_k`. The
objective is

`0.5 * d_k.T @ R_u @ d_k + rho_s * w_s.T @ s_k
 + 0.5 * eta_s * (w_s.T @ (s_k ** 2))`

and, for groups 2 and 3, also includes the corresponding linear and quadratic
penalties on protection slack `xi_k`. `R_u` is constructed from the configured
control correction weight and `optimization.control_weights`.

For the currently violated selected rows, the linearized soft constraints are
`r_k + J_k @ d_k <= s_k`, with `0 <= s_k <= r_k`. The slack is both penalized
and upper bounded by the current positive residual. For all selected rows of
previous groups, the protection constraints are `H_prev @ d_k <= xi_k`, with
`0 <= xi_k <= projection.protection_tolerance`. Control magnitude bounds are
hard constraints in the hard-box restoration QP and every priority-group QP.
There is no group-level minimum-improvement-ratio constraint and no QP Trust
Region. Nonlinear Backtracking and Stage-2 acceptance remain authoritative.

## Solver diagnostics

Each solved group records OSQP status, iterations, primal and dual residuals,
solve time, state slack, protection slack, and linearized improvement. Solver
failure is reported in the JTP diagnostics and is never redirected to LS.

## Stage-3 road active-set sampling

The full road residual remains evaluated at all five footprint-circle centers
for all 40 future steps. The active candidate pool uses the center circle at
every second future step, with a global Top-K cap of 30. Full-resolution road
rows remain in nonlinear acceptance and final feasibility metrics. Collision
geometry and active sets are unchanged.

## API

`JointTrajectoryProjector(execution_model, config).project(nominal_controls,
context, road_geometry, external_actors=()) -> JTPResult` and
`evaluate_preprojection(...)` retain their existing signatures. No checkpoint,
K2 network, training code, or evaluation interface is changed.

## Tests

`python -m pytest -q tests/test_joint_trajectory_projection.py tests/test_jtp_grouped_stage1.py tests/test_jtp_stage3_road_active_set.py tests/test_jtp_qp.py`
