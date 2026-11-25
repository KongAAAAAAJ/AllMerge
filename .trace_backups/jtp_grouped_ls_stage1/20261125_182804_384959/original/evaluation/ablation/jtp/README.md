# TRACE Ablation 3: Joint Trajectory Projection

This evaluation compares **Baseline**, the raw output of one frozen control
Diffusion/K2 inference, with **Ours**, the result returned by the same inference
after applying the current Active-set Weighted Joint Jacobian Projection (JTP).
No checkpoint is retrained, and Baseline/Ours use identical nominal controls
within each synchronized frame and inference seed.

## Commands

Run from the `all_merge/` worktree root:

```bash
# Smoke test
python evaluation/ablation/run_jtp.py \
  --checkpoint outputs/trace_control_koopman_d2/best.pt \
  --num-frames 3 --seeds 0

# Formal multi-seed offline evaluation
python evaluation/ablation/run_jtp.py \
  --checkpoint outputs/trace_control_koopman_d2/best.pt \
  --num-frames 100 --seeds 0,1,2,3,4 \
  --split test

# Verify that partial JTP iterates are retained and exceptions are not hidden
python -m pytest -q tests/test_jtp_ablation.py
```

Use `--config configs/evaluation/jtp_ablation.yaml` or pass different paths
with `--dataset-root`, `--jtp-config`, `--diffusion-config`, and `--output-dir`.
`--num-frames 0` uses all complete synchronized frames; the default selects 100
complete frames with a fixed sampling seed. The dataset must include
`episode_index`, `frame_index`, `ego_index`, synchronized K2 history, and paired
control anchors. Incomplete multi-truck frames are excluded by the shared
complete-frame selector.

## Projection method and output contract

- `joint_projector.py`: active-set, violation-weighted minimum-norm Jacobian
  projection in the joint control space of all trucks. The local metric
  approximates trajectory deviation under the frozen Koopman execution map.
- `constraints.py` and `geometry.py`: full-horizon yaw-rate, sideslip, roll,
  LTR, road-boundary, curvature, and control-magnitude residuals.
- External-actor and inter-truck collision avoidance are handled upstream by
  Adaptive Anchor scoring/selection. Control-rate constraints are not included
  in the current lightweight JTP update.
- Each sample always uses the controls and Koopman rollout returned by JTP.
  The evaluator never replaces an infeasible or partially improved JTP result
  with the nominal Diffusion output. If projection raises an exception or the
  road geometry is invalid, evaluation stops and reports the error; the sample
  is not silently skipped or relabeled as a JTP result.
- `projector_constraints_satisfied` is only a diagnostic flag for the constraint
  families implemented inside JTP. It is not the paper-level joint feasibility
  metric. A returned JTP result can be partially improved while this flag is
  false. The algorithm's best accepted iterate is retained.
- `projection_applied` means that a vehicle's normalized control correction
  exceeds `projection.change_tolerance`. A zero correction can occur when the
  nominal output already satisfies the projector's internal constraints or
  when no candidate update improves the projection merit.

## Paper metrics and supplementary diagnostics

The four finalized paper metrics remain unchanged:

1. **Joint Feasibility (%)**: percentage of synchronized N-truck frame/seed
   plans for which every controlled vehicle satisfies all independently
   evaluated stability, geometry, collision, control-magnitude, and
   control-rate constraints over the full prediction horizon. Each complete
   formation plan counts once.
2. **Geometric Violation (%)**: percentage of individual truck trajectories
   with any road-boundary or curvature violation over the full horizon.
3. **Stability Violation (%)**: percentage of individual truck trajectories
   with any yaw-rate, sideslip, roll-angle, or LTR violation over the full
   horizon.
4. **Mean Trajectory Deviation (m)**: average over vehicles of the mean
   pointwise XY distance at `t=1..H` between the JTP execution trajectory and
   that same sample's nominal K2 execution trajectory. Baseline is N/A.

The method change from SCP-QP to Jacobian projection does **not** change these
outcome metrics. The obsolete **Projection/Solver Success Rate is removed**:
the current method does not invoke a constrained optimization solver, and
its internal constraint flag is not equivalent to full-plan feasibility.

The CSV and JSON outputs additionally report collision and control-magnitude/
control-rate violation rates, normalized constraint-excess magnitudes,
projection-applied rate, normalized control-correction magnitude, projection
latency, and projection iterations. These diagnostics quantify partial
improvements even when a sample remains infeasible; they are not substitutes
for the four primary paper metrics.

## Evaluation protocol

- One complete synchronized N-truck frame is the statistical unit.
- Baseline and Ours share the exact same raw Diffusion inference for each
  frame/seed pair.
- Stability and geometric constraints are checked for every future step of
  every vehicle. Inter-truck collision is checked jointly across the formation;
  external collision is checked against constant-velocity actor predictions.
- Paired confidence intervals use cluster bootstrap over physical
  episode/frame clusters, preserving all inference seeds and trucks within
  each sampled cluster.
- The evaluation is offline expert-data open loop. It does not measure
  TruckSim closed-loop task success, executed spacing, fuel, or physical
  closed-loop stability.

## Outputs

- `jtp_per_vehicle.csv`: paired per-vehicle outcomes and projection diagnostics.
- `jtp_per_seed.csv`: per-seed summaries.
- `jtp_summary.csv`: aggregate metrics for Baseline and Ours.
- `jtp_paired_statistics.json`: paired differences and cluster-bootstrap CIs.
- `jtp_projection_diagnostics.json`: per-frame projection status, internal
  constraint status, correction magnitudes, iteration counts, and latency.
- `jtp_solver_summary.json`: termination status, active-set truncation, and
  group-wise residual/count/L2 progress for stability, road, curvature, control
  magnitude, inter-truck collision, external collision, and control rate.
- `projection_ablation_table.tex`: four-metric paper table.
- `projection_ablation.pdf` / `.png`: seven-panel trajectory, geometry, and
  four-channel chassis-response plot.
- `projection_ablation_figure_data.npz`: pickle-free data for figure regeneration
  with `python evaluation/ablation/plot_jtp_figure.py`.
- `jtp_manifest.json`: checkpoint/config hashes, metric definitions, and
  evaluation provenance.

Numerical limits in `configs/evaluation/jtp_ablation.yaml` and
`configs/projection/jtp.yaml` must be checked and frozen before publication.
