# TRACE Ablation 3: Joint Trajectory Projection

This evaluation compares **Baseline**, the raw output of one frozen control
Diffusion/K2 inference, with **Ours**, the result returned by the same inference
after applying priority-grouped sequential JTP with the Stage-1 damped LS backend.
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
python -m pytest -q tests/test_joint_trajectory_projection.py tests/test_jtp_grouped_stage1.py
```

Use `--config configs/evaluation/jtp_ablation.yaml` or pass different paths
with `--dataset-root`, `--jtp-config`, `--diffusion-config`, and `--output-dir`.
`--num-frames 0` uses all complete synchronized frames; the default selects 100
complete frames with a fixed sampling seed. The dataset must include
`episode_index`, `frame_index`, `ego_index`, synchronized K2 history, and paired
control anchors. Incomplete multi-truck frames are excluded by the shared
complete-frame selector.

## Projection method and output contract

- `joint_projector.py`: priority-grouped outer nonlinear projection. `linearization.py`
  shares structured K2 sensitivities and `grouped_solver.py` uses damped LS plus
  verified linear cross-group protection. The QP backend is reserved for Stage 2.
- All seven constraint families (including inter-truck/external collision and
  control-rate residuals) participate in projection and independent evaluation.
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
  when no candidate passes nonlinear group-wise acceptance.

## Paper metrics and supplementary diagnostics

Independent evaluation excludes the history-to-first-prediction control-rate
transition. Acceleration and steering rates, their violation flags, and normalized
excess statistics use only `(u[t] - u[t-1]) / dt` at prediction steps 2..H.
This applies to all summaries, joint feasibility, paired statistics, and tables.
The projector and its internal diagnostics still include the historical transition.
For H=1, rate violation flags are false, excess statistics are zero, and rate
distributions have no samples. Distribution metadata v2 records this scope;
legacy files retain their original semantics when replotted. Use a fresh output
directory when reevaluating; existing results are not migrated automatically.

`--statistics-unit {both,trajectory,step}` defaults to `both`. Each selected
unit has its own directory and five-metric table. The following definitions
describe trajectory statistics; step statistics use only steps 2..H for every
constraint and XY deviation, with curvature assigned to its interval end node.
Joint feasibility then requires all trucks to be feasible at that same step.
The additional **Feasibility (%)** column counts feasible individual vehicles
(whole trajectories or individual steps). Empty step reports (H=1) use null/N/A.

The trajectory metrics are:

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

The method change to priority-grouped LS does **not** change these
outcome metrics. The obsolete **Projection/Solver Success Rate is removed**:
the current method does not invoke a constrained optimization solver, and
its internal constraint flag is not equivalent to full-plan feasibility.

The CSV and JSON outputs additionally report collision and control-magnitude/
control-rate violation rates, normalized constraint-excess magnitudes,
projection-applied rate, normalized control-correction magnitude, projection
latency, and projection iterations. These diagnostics quantify partial
improvements even when a sample remains infeasible; they are not substitutes
for the primary paper metrics. Detailed normalized excess and solver diagnostics
are retained in trajectory reports; step reports expose per-step violation flags.

## Evaluation protocol

Frame/seed pairs run serially by default (`--workers 1`). Use `--workers 2
--worker-threads 1` for isolated spawn processes, each loading its own model
once and sharing the selected device. The trucks in each frame remain jointly
projected. Submission is bounded, results are merged in original order, and a
failed task aborts the evaluation. Single-GPU speedup is workload-dependent;
concurrent per-frame latency includes resource contention. Manifest `execution`
records evaluation wall time and throughput for comparison. Final reporting
and plotting remain in the parent process.

- Statistical units are selected by the CLI; bootstrap always resamples physical
  episode/frame clusters, preserving all seeds, trucks, and steps within a cluster.
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

- `<unit>/jtp_per_vehicle_<unit>.csv`: paired trajectory or vehicle-step records.
- `<unit>/jtp_per_seed_<unit>.csv`: per-seed summaries.
- `<unit>/jtp_summary_<unit>.csv`: aggregate metrics for Baseline and Ours.
- `<unit>/jtp_paired_statistics_<unit>.json`: differences and cluster-bootstrap CIs.
- `jtp_projection_diagnostics.json`: per-frame projection status, internal
  constraint status, correction magnitudes, iteration counts, and latency.
- `jtp_solver_summary.json`: termination status, grouped solver timing, protection, cache reuse, active-set truncation, and
  group-wise residual/count/L2 progress for stability, road, curvature, control
  magnitude, inter-truck collision, external collision, and control rate.
- `<unit>/projection_ablation_table_<unit>.tex`: five-metric paper table with
  Joint Feas., individual Feas., geometry/stability violation, and deviation.
  Old unsuffixed statistical files are no longer written; historical files are
  not deleted. Use a fresh output directory to avoid mixing report versions.
- `projection_ablation.pdf` / `.png`: seven-panel trajectory, geometry, and
  four-channel chassis-response plot.
- `projection_ablation_figure_data.npz`: pickle-free data for figure regeneration
  with `python evaluation/ablation/plot_jtp_figure.py`.
- `jtp_manifest.json`: checkpoint/config hashes, metric definitions, and
  evaluation provenance.

Numerical limits in `configs/evaluation/jtp_ablation.yaml` and
`configs/projection/jtp.yaml` must be checked and frozen before publication.
