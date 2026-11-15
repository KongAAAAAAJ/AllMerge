# JTP update notes (2026-11-15)

## Projection output contract
- The active implementation is `highway_env/planner/diffusion/projection/joint_projector.py`.
- The method is Active-set Weighted Joint Jacobian Projection. It does not call an SCP-QP constrained optimizer.
- `JTPResult.success` was replaced by `JTPResult.projector_constraints_satisfied`. This flag covers only the constraint families implemented inside JTP and is not the paper-level full-platoon feasibility metric.
- Offline evaluation always retains `JTPResult.controls` and its returned Koopman rollout, even when the JTP-internal constraint flag is false. Projection exceptions and invalid road geometry abort evaluation; samples are not silently skipped or replaced by nominal Diffusion outputs.
- Runtime control execution also consumes every returned JTP control sequence. A projector-internal infeasibility status does not route control back to raw Diffusion or Polynomial. The initial K2 history warmup remains necessary before JTP can run.

## Evaluation metrics
The four finalized paper metrics remain unchanged:
1. Joint Feasibility Rate, evaluated per synchronized N-truck frame/seed against all full-horizon stability, geometry, collision, control-magnitude, and control-rate checks.
2. Geometric Violation Rate, per vehicle trajectory.
3. Stability Violation Rate, per vehicle trajectory.
4. Mean Trajectory Deviation, per vehicle, relative to the same sample's nominal K2 execution trajectory.

The obsolete Projection/Solver Success Rate was removed. Supplementary diagnostics now include projection-applied rate, normalized control correction, normalized stability/geometry/control excess magnitudes, collision penetration, projection latency, and projection iterations.

## Removed obsolete code
- Removed archived SCP-QP `scp_solver.py` and `qp_builder.py` implementations from the old JTP backup snapshots.
- Renamed `joint_solver.py` to `joint_projector.py`.
- Removed the obsolete single-vehicle validator that depended on the removed `ExternalTraffic` projector API.
- Removed the `--require-paper-solver` and `--allow-fallback` evaluation switches and their associated capability/fallback reporting.

## Validation
- All active Python files passed syntax compilation.
- `python -m pytest -q tests/test_jtp_ablation.py`: 2 passed.
