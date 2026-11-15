# TRACE Ablation 3: Joint Trajectory Projection

This sibling package to `evaluation/ablation/anchor_ablation/` and
`evaluation/ablation/fuel_aware/` evaluates **Baseline = one frozen control
Diffusion/K2 rollout**, and **Ours = JTP applied to that *same* selected
control sequence and execution trajectory**. No checkpoint is retrained, and
there is no separately seeded second inference for Ours.

## Commands

Run from the `all_merge/` worktree root:

```bash
# Smoke test after placing the control K2 checkpoint and expert dataset at the configured paths
python evaluation/ablation/run_jtp.py \
  --checkpoint outputs/trace_control_koopman_d2/best.pt \
  --num-frames 3 --seeds 0

# Formal multi-seed offline analysis. Check limits first.
python evaluation/ablation/run_jtp.py \
  --checkpoint outputs/trace_control_koopman_d2/best.pt \
  --num-frames 100 --seeds 0,1,2,3,4 \
  --split test

# Reject a solver that does not implement the full Section-V method
python evaluation/ablation/run_jtp.py \
  --checkpoint outputs/trace_control_koopman_d2/best.pt \
  --require-paper-solver

python -m pytest -q tests/test_jtp_ablation.py
```

Use `--config configs/evaluation/jtp_ablation.yaml` or pass different paths
with `--dataset-root`, `--jtp-config`, `--diffusion-config`, `--output-dir`.
`--num-frames 0` uses all complete synchronized frames; default picks 100
complete frames with a fixed sampling seed. The shared dataset is required to
include `episode_index`, `frame_index`, `ego_index`, synchronized K2 history,
and paired control anchors. No mixing of incomplete multi-truck frames.

## Source-to-paper contract

- `runner.py`: one Diffusion selected-mode inference per frame/seed,
  identical nominal controls and world-frame K2 execution in both arms;
  synchronized N-truck JTP; exact fallback to nominal on solver failure.
- `data.py`: reuses the existing Anchor evaluation complete-frame selector;
  constructs same-road road corridor with full truck footprint; collects
  noncontrolled surrounding actors across all controlled trucks; constant
  world velocity and fixed yaw (the same CV assumption as Anchor scoring).
- `metrics.py`: independently checks every step's yaw, beta, roll, LTR;
  road/circle and curvature; full-circle pairwise intertruck and external
  collision; acceleration/steering magnitude and **rate**, including the
  previous observed control node. Collision checks include future samples
  `t=0.1..4.0 s`; the initial t=0 pose is not controllable.
- `reporting.py`: IEEE two-row paper table, per vehicle/seed CSVs, status and
  solver diagnostics, paired deltas and cluster bootstrap confidence intervals
  resampled by *physical* episode/frame (all seeds/trucks preserved).
- `visualization.py`: six plots in the exact paper order: (a) global XY,
  (b) detail and correction arrow, (c) curvature, (d) yaw rate,
  (e) roll angle, (f) LTR. The scene is fixed-random, **not** selected for
  largest improvement.

The metric **denominators are explicitly separated**:

- Joint Feas. (%): percent of synchronized **N-truck frame/seed plans** for
  which *all trucks jointly* meet every stability, geometry, collision,
  control and rate constraint; one joint plan counts once.
- Geometric Viol. (%): percent of individual truck trajectories with any
  road/curvature violation.
- Stability Viol. (%): percent of individual truck trajectories with any
  yaw/beta/roll/LTR violation.
- Deviation (m): average over vehicles of mean t=1..H Euclidean XY deviation
  from *that same* original K2 execution trajectory. Baseline is N/A.

Other columns (collision and control rates, success/fallback, runtime) remain
in CSV and summary for traceability, but are not added to the four-column
paper table.

## Critical solver compatibility warning

**As inspected in `AllMerge-main (3).zip` on 2026-11-15:** the current
`projection/joint_solver.py` implements an active-set weighted Jacobian
correction, explicitly *not* the final paper's SCP joint quadratic program.
Its own `success` check excludes collision and control-rate constraints.
Moreover, `evaluation/validate_jtp_single_vehicle.py` currently imports an
`ExternalTraffic` symbol absent from `projection/types.py`. This new
self-contained ablation avoids that broken script.

Therefore **the baseline and Ours CSV metrics are honest full-constraint
measurements of the current implementation**, but the output
`projection_ablation_table.tex` and PDF are *watermarked DIAGNOSTIC ONLY*
until an updated JTP solver explicitly advertises all capabilities via
`JointTrajectoryProjector.PAPER_JTP_CAPABILITIES`, and independent tests verify
those behaviors. `--require-paper-solver` raises immediately if any are
missing. The manifest also lists the missing capabilities.

## Numerical parameters

The 2026-11-15 source does not supply calibrated experimental values for
`collision_clearance_m`, `acceleration_rate_max_mps3`, or
`steering_rate_max_rad_s`. The evaluation YAML contains **explicit provisional
engineering defaults** solely for diagnostics; **they are not claimed to be
values from the finalized paper**. Review and freeze them alongside the JTP
stability/geometry/control YAML before collecting publication-quality results.
The default geometry/vehicle assumptions also require confirming the road
map representation and footprint dimensions for the actual dataset.

The module does not edit `anchor_ablation`, `fuel_aware`, the trained model,
Koopman weights, the JTP solver, or any runtime inference behavior.
