# TRACE Ablation 3: Joint Trajectory Projection

This evaluation compares Baseline, the raw output of one frozen control Diffusion/K2 inference, with Ours, the result returned by JTP. JTP splits each synchronized frame into independent single-vehicle projection problems. Each vehicle optimizes its own 80-dimensional control vector; the three vehicles are not jointly optimized.

## Commands

Run from the `all_merge/` worktree root:

```bash
python evaluation/ablation/run_jtp.py \
  --checkpoint outputs/trace_control_koopman_d2/best.pt \
  --num-frames 3 --seeds 0

python evaluation/ablation/run_jtp.py \
  --checkpoint outputs/trace_control_koopman_d2/best.pt \
  --num-frames 100 --seeds 0,1,2,3,4 --split test

python -m pytest -q tests/test_joint_trajectory_projection.py
```

Use `--config configs/evaluation/jtp_ablation.yaml` or pass different paths
with `--dataset-root`, `--jtp-config`, `--diffusion-config`, and `--output-dir`.

## Projection method

- `joint_projector.py` preserves the batch runtime interface and executes one independent 80-D projection per controlled vehicle.
- `projection.solver_method` selects `damped_least_squares` or `constrained_qp`.
- Damped LS retains active-set/Top-K selection, damping, bounded Cholesky active-set solving, Trust Region, and metric half-space protection.
- Constrained QP retains OSQP Full-Group QP over all rows in the sparse temporal pool, with state slack, common tolerance handling, cross-group protection, and hard control bounds.
- Both backends use fixed priority groups: G1 stability, G2 road/curvature, G3 control-rate.
- `projection.constraint_sampling_stride` controls initial G1/G2 sampling: 1/2/4 correspond to 40/20/10 time points. G3 and control magnitude bounds remain full horizon. Stage 2 checks all 40 time points and adaptively adds unsampled G1/G2 violation times.
- Inter-truck and external-actor collision constraints are not part of projection. Collision metrics remain independent full-horizon evaluation metrics.

## Evaluation metrics

Trajectory-level reports retain the existing joint-feasibility, individual-vehicle feasibility,
geometric-violation, stability-violation, and mean trajectory-deviation metrics.

The step-level paper table reports the satisfaction rate of each displayed constraint,
overall constraint feasibility, and per-vehicle projection latency:

- **Geometry:** road-boundary and curvature satisfaction.
- **Stability:** yaw-rate, sideslip-angle, roll-angle, and lateral-load-transfer-ratio (LTR) satisfaction.
- **Control:** acceleration magnitude, steering magnitude, acceleration-rate, and steering-rate satisfaction.
- **Overall feasibility:** individual vehicle-step feasibility and synchronized platoon-step feasibility.
- **Real-time:** mean and P95 latency of one independently timed single-vehicle projection.

Node-state constraints use prediction steps 1..H. Curvature and control-rate constraints
are interval quantities and use steps 2..H. Each individual constraint satisfaction rate
uses only its valid samples. Collision constraints are omitted as separate table rows but
remain included in individual vehicle-step and synchronized platoon-step feasibility.

Individual vehicle-step feasibility requires all applicable constraints, including collision
checks, to be satisfied by one vehicle at one prediction step. Synchronized platoon-step
feasibility requires all vehicles to satisfy all applicable constraints at the same prediction
step. Curvature and control rates are not applicable at step 1.

The latency mean and P95 are computed from one independently measured `_project_single`
call per vehicle. The serial three-vehicle `project()` wall time remains a frame diagnostic
and is not used as the single-vehicle latency statistic.

The independent evaluator excludes the history-to-first-prediction control-rate transition.
Projection itself includes the previous applied control to step 1. Baseline and Ours use the
same nominal controls, K2 model, road geometry, and external-actor evaluation set.

The evaluation is offline expert-data open loop. It does not measure TruckSim closed-loop
task success, executed spacing, fuel, or physical closed-loop stability.
