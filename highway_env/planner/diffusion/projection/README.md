# TRACE Joint Trajectory Projection (JTP)

JTP supports two priority-grouped solver backends selected by
`configs/projection/jtp.yaml`:

- `damped_least_squares`: the Stage-3 Cholesky grouped-LS backend, including
  its fixed control box, cumulative Trust Region, active-row selection, and
  metric half-space protection.
- `constrained_qp`: three sequential OSQP subproblems. Each QP includes every
  residual row belonging to its current priority group, including violated
  rows omitted by the LS Top-K active set and currently satisfied rows. Every
  row of all preceding groups is included in explicit cross-group protection.
  The QP backend has no Trust Region.

When the nominal controls violate the magnitude box, QP first solves a separate
control-only restoration QP. It then solves the three priority groups. Both
backends share the six state-constraint families, priority ranking, K2
Jacobian construction, road-association freezing, Backtracking, and Stage-2
full-resolution nonlinear acceptance. Selecting QP requires the optional
dependencies installed with `pip install -e ".[jtp]"`. There is no automatic
solver fallback.

## QP formulation

For priority group `k`, the incremental control correction is `d_k`. The
objective is

`0.5 * d_k.T @ R_u @ d_k + rho_s * w_s.T @ s_k`
`+ 0.5 * eta_s * (w_s.T @ (s_k ** 2))`

and, for groups 2 and 3, also includes the corresponding linear and quadratic
penalties on protection slack `xi_k`. `R_u` is constructed from the configured
control correction weight and `optimization.control_weights`.

All rows of the current group are included. Let `r0` be the group's residual
relative to its outer-iteration baseline and `J` its Jacobian. The residual at
the current sequential iterate is `r = r0 + J @ p`, where `p` is the
accumulated correction from preceding groups. The current-group constraint is
`r + J @ d_k <= s_k`. Its slack upper bound is `r0 + nondegradation_tolerance`
for rows that were violated at the outer-iteration baseline, and zero for rows
that were satisfied. Thus an originally satisfied row cannot become newly
violated, while an originally violated row cannot regress beyond the Stage-2
per-row tolerance. Every state slack is nonnegative, upper bounded, and has
linear and quadratic penalties.

For all rows of preceding groups, the protection model is referenced to the
same outer-iteration baseline. Previously violated rows may increase only
within the smaller of `protection_tolerance` and
`nondegradation_tolerance`; previously satisfied rows must remain satisfied.
Protection slacks are nonnegative, upper bounded, and linearly/quadratically
penalized. Control magnitude bounds are hard constraints in the hard-box
restoration QP and every priority-group QP. There is no group-level
minimum-improvement-ratio constraint and no QP Trust Region. Nonlinear
Backtracking and Stage-2 acceptance remain authoritative.

## Constraint scope by backend

- LS uses the configured active-set selection, including Stage-3 road
  candidate sampling and per-family Top-K limits.
- QP does not call the active-set selector and does not use
  `max_active_per_group`. All residual rows already constructed by the JTP
  constraint model enter their corresponding priority-group QP. The existing
  road residual construction and center-circle/every-two-step candidate logic
  are not changed; the latter is used only by the LS active-set path.
- Stage-2 nonlinear acceptance continues to evaluate every row for both
  backends.

## Solver diagnostics

Each solved group records its total constraint-row count, initially violated
and inactive row counts, OSQP status, iterations, primal and dual residuals,
solve time, state slack, protection slack, and linearized improvement. Solver
failure is reported in JTP diagnostics and is never redirected to LS.

## API

`JointTrajectoryProjector(execution_model, config).project(nominal_controls,
context, road_geometry, external_actors=()) -> JTPResult` and
`evaluate_preprojection(...)` retain their existing signatures. No checkpoint,
K2 network, training code, or evaluation interface is changed.

## Tests

`python -m pytest -q tests/test_joint_trajectory_projection.py tests/test_jtp_grouped_stage1.py tests/test_jtp_qp.py`
