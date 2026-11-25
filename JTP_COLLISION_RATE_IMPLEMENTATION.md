# JTP Collision and Control-Rate Constraint Extension

This incremental update extends the Active-set Weighted Joint Jacobian Projection with three additional residual families:

- **Inter-truck collision:** pairwise footprint-circle separation constraints across all controlled trucks in the synchronized projection batch.
- **External-actor collision:** footprint-circle separation against constant-velocity world-frame predictions of non-controlled road actors.
- **Control rates:** first differences of longitudinal acceleration and front-wheel steering, with the first difference measured against the last applied control in synchronized K2 history.

The independent evaluator recomputes these constraints using the same configured clearance, rate limits, and tolerances. The four paper-level JTP metrics remain unchanged: joint feasibility, geometric violation, chassis-stability violation, and mean trajectory deviation. Collision and control-rate violation rates remain supplementary diagnostics.

The projection remains a local nonlinear Jacobian method. Adding these constraint families does not guarantee global feasibility. A sample can remain infeasible because of conflicting constraints, active-set truncation, limited iterations, poor local Jacobian conditioning, or insufficient control authority. Inspect `jtp_solver_summary.json` and per-sample diagnostics before changing thresholds or weights.

## Installation behavior

The projection README patch is optional because local worktrees can contain documentation revisions that differ from the uploaded project snapshot. If no exact semantic anchor matches, the installer skips only that README update and installs the mandatory code/config patches after all remaining checks pass. It does not force or overwrite unmatched documentation. This file provides the corresponding implementation notes.
