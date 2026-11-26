# TRACE Joint Trajectory Projection (Stage 2)

The runtime projection algorithm is priority-grouped damped least squares with
fixed hard control bounds, cross-group protection, and outer nonlinear
validation. The six state-constraint families are stability, road, curvature,
inter-truck collision, external-actor collision, and control rate. Acceleration
and steering magnitude limits form a separate fixed hard constraint and are
not included in priority scoring or state active-set caps.

## Solver configuration

`configs/projection/jtp.yaml`, `projection.solver_method: damped_least_squares`
is the implemented Stage-2 backend. `constrained_qp` remains reserved for a
later solver study and selecting it raises `NotImplementedError`. `group_count`
is 2 or 3. The six state families are ranked by
`q * (1 + max(relu(normalized_residual - tolerance)))`, split into near-equal
groups, and fixed for one `project()` call. Control magnitude bounds are not
ranked and are imposed in every group subproblem.

`constraint_tolerance` is the common normalized tolerance for state families.
Control-rate SI tolerance is converted to its normalized channel tolerance by
the same threshold builder used for active-row selection and nonlinear
metrics. Control magnitude bounds have zero feasibility slack.

## Execution

- `linearization.py`: cache conditional A/B, z0, interpolation, and analytic
  latent sensitivities across the call. Use six batched pointwise decoder VJPs
  and trapezoidal pose sensitivity. Recompute output/constraint derivatives at
  the outer working point; freeze road association during linearization only.
  The callable adapter contract uses one shared autograd output Jacobian. An
  empty state active set still builds the objective metric for hard-box repair.
- `grouped_solver.py`: reuse one objective-metric Cholesky factorization. First
  find the metric-minimum correction that enters the hard control box when the
  nominal controls are outside it. Every state-group LS subproblem then solves
  under the same final-control box and cumulative normalized trust box. The
  unconstrained step uses the weighted Gram Cholesky formulation; trust overflow
  triggers a small number of internal damping retries, followed by a reduced
  primal Cholesky active-set solve for remaining box-active variables. Previous
  selected rows are protected by metric half-space projection while retaining
  the current box. No post-solve elementwise clipping or scalar trust scaling is
  applied.
- `joint_projector.py`: nonlinear backtracking evaluates the unmodified
  candidate controls. Road associations are frozen within one line search and
  refreshed after an accepted candidate. A candidate must satisfy the hard
  control box; every state-constraint row must have non-increasing tolerance
  excess, and each priority group must have a non-increasing violation count.
  At least one state constraint must improve materially, unless the candidate
  restores an initially infeasible hard control box without state regression.

The road model, five-circle footprint, collision model, K2 execution model,
independent paper metrics, and evaluation interfaces remain unchanged.
Road active-set sampling and its cap remain unchanged in Stage 2; road active-set
reduction is reserved for Stage 3.

## API

`JointTrajectoryProjector(execution_model, config).project(nominal_controls,
context, road_geometry, external_actors=()) -> JTPResult` and
`evaluate_preprojection(...)` retain their existing signatures. No checkpoint,
K2 network, training code, or evaluation metric modification is required.

## Tests

`python -m pytest -q tests/test_joint_trajectory_projection.py tests/test_jtp_grouped_stage1.py`
