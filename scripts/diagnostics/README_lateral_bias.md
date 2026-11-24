# TRACE lateral-bias diagnostics (read-only)

These experiments do not edit training code, YAML configurations, model weights,
or dataset shards. Run from the repository root.

## 1. Dataset distribution and endpoint-bias statistics

```bash
python scripts/diagnostics/analyze_lateral_bias.py \
  --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k \
  --split val \
  --output-dir outputs/diagnostics/lateral_bias/val
```

This computes:

- Signed target-lane curvature distribution from `target_lane_polyline[..., 0:2]`.
- Expert terminal lateral displacement from `future_trajectory_dense[-1, 1]`.
- Target-point versus expert-endpoint lateral offset.

Add `--checkpoint <path-to-best.pt>` to also run the deployed `infer_multimodal`
path and record selected-mode terminal lateral position, lateral error, FDE,
and selected-mode histogram. The checkpoint is loaded read-only.

Run the same command on `train`, `val`, and `test` to distinguish dataset
imbalance from generalization effects. Use the same checkpoint, seed and sample
limits when comparing splits.

## 2. Same-control execution-model comparison

First choose one raw sample and generate a deterministic Diffusion control
sequence using the checkpoint's deployed Candidate Scoring selector:

```bash
python scripts/diagnostics/compare_execution_models.py \
  --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k \
  --split test --sample-index 0 \
  --checkpoint outputs/anchor_ablation/safety_diffusion_ckpt/A1/best.pt \
  --seed 2048 \
  --output-dir outputs/diagnostics/execution_models/sample_0
```

This always compares Kinematic Bicycle and Koopman K2. To include the native
TruckSim plant, run on the Windows machine with the matching TruckSim model and
DLL:

```powershell
python scripts/diagnostics/compare_execution_models.py `
  --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k `
  --split test --sample-index 0 `
  --checkpoint outputs/anchor_ablation/safety_diffusion_ckpt/A1/best.pt `
  --trucksim-simfile "D:\\...\\Lx_MUy.sim" `
  --trucksim-dll "D:\\...\\Lx_MUy.dll" `
  --collect-config configs/koopman/collect.yaml `
  --expected-payload-kg 0 --expected-road-mu 0.8 `
  --output-dir outputs/diagnostics/execution_models/sample_0
```

The payload and friction flags are explicit operator confirmations that the
selected `.sim` is configured for the same physical conditions as the raw
sample. They do not modify or inspect the physical `.sim` setup. Confirm the
payload and road friction in TruckSim before interpreting the result.

Alternatively, pass `--controls-npy path/to/controls.npy` containing exactly
`[40, 2]` values ordered as `[ax_mps2, front_wheel_delta_rad]` to compare a
previously saved control sequence. `--mode-index` selects a specific generated
candidate; omit it to use the deployed Candidate Scoring selection.

Outputs:

- `input_controls.csv`: the exact shared 40-step reference sequence.
- `summary.json`: terminal lateral displacement, terminal yaw, expert-relative
  lateral error, and curvature summaries.
- `trajectories.csv`: aligned 10 Hz trajectories, yaw, curvature and lateral
  error for each available model.
- `execution_comparison.png`: trajectory, yaw response, lateral error and
  curvature plots.
- `trucksim_raw_exports.csv`: native-rate raw TruckSim channels when TruckSim
  is enabled.

TruckSim is a one-shot native DLL run. The script follows the existing project
workaround and skips native termination because the current local DLL has a
known teardown crash. Start a fresh Python process for each TruckSim run.

## Interpretation notes

- All three plants receive the same 40-step reference `[ax, delta_front]`.
- TruckSim receives the reference through the project's calibrated
  `ReferenceActuator` (engine map, longitudinal PID, steering ratio and lateral
  controller), not by treating acceleration as throttle or road-wheel angle
  as steering-wheel angle.
- Kinematic Bicycle and Koopman K2 use their native project implementations.
- TruckSim outputs are transformed from its world `Xo/Yo/Yaw` exports into the
  planning-time ego-local frame. Export order/units must match the active
  19-channel TruckSim model documented by the project.
- This is an open-loop plant comparison, not a closed-loop planner evaluation.
