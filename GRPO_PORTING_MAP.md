# GRPO_PORTING_MAP

Baseline target: AllMerge `63a7365` or a descendant compatible `main`.

## 1. Frozen AllMerge trunk

The following files are integration dependencies and are **not** modified by this migration:

- `highway_env/planner/diffusion/config.py`
- `highway_env/planner/diffusion/tensor_adapter.py`
- `highway_env/planner/diffusion/encoders.py`
- `highway_env/planner/diffusion/diffusion_schedule.py`
- `highway_env/planner/diffusion/mode_assignment.py`
- `highway_env/planner/diffusion/structured_model.py`
- `highway_env/planner/diffusion/runtime.py`

## 2. Source → target map

| Capability | Diffusion-metadrive source to inspect first | Action | AllMerge target | Notes |
|---|---|---|---|---|
| DDIM transition + log-prob | `models/diffusion/*scheduler*`, DiffusionDriveV2 `DDIMScheduler_with_logprob` lineage | ADAPT | `grpo/scheduler.py` | Reuse AllMerge `alpha_cumprod`; preserve clean-x0 prediction semantics. |
| Sampling trace / replay | `models/diffusion/*`, selector refinement path | ADAPT | `grpo/sampling.py` | Calls existing `scene_encoder`, `denoiser`, adapter and schedule. |
| Group-relative advantage | existing GRPO/refinement code | COPY-equivalent | `grpo/objective.py` | Standard group normalization; no critic. |
| Ratio / clip | existing GRPO/refinement code | COPY-equivalent | `grpo/objective.py` | PPO-style clipped surrogate on diffusion transition log-probs. |
| Current/reference policy | selector refinement trainer | ADAPT | `grpo/trainer.py` | Deep-copy fixed reference; no second planner implementation. |
| Trainable diffusion head | old `diff_decoder` | ADAPT | `denoiser.layers`, `denoiser.reg_head` | Scene encoder and classifier remain frozen by default. |
| Reward callback | old reward callback | DROP/REPLACE | `grpo/reward_adapter.py` | Delegates to W4 `evaluate_candidates()`; reward math is not copied into trainer. |
| `PlatoonEnv.get_state/set_state` | closed-loop executor / branch rollout | ADAPT | `grpo/rollout_adapter.py` | Native snapshot if available; deepcopy fallback for correctness. |
| Checkpoint | old trainer checkpoint | ADAPT | `grpo/checkpoint.py` | Accepts W2 `state_dict` / `model_state_dict` / raw state dict. |
| Training entry | `train/train_selector.py`, GRPO train entry | ADAPT | `train_grpo.py` | Reuses W1 feature contract and W2 checkpoint. |
| Diagnostics | existing GRPO diagnostics | DEFER | trainer metrics | First restore ratio/KL/clip/grad/reward; ChassisFusion diagnostics only after 100-step gate. |
| Risk-PACT | Risk-PACT branch | DROP FOR W3 | none | Explicitly deferred until GRPO correctness is proven. |

## 3. Why scheduler code is ADAPT, not blind COPY

AllMerge's current planner predicts the clean sample `x0` directly and owns a lightweight `TruncatedDDIMSchedule`. The old DiffusionDrive/Diffusion-metadrive lineage also contains replayable DDIM log-probability math, but the old scheduler object and surrounding planner interfaces should not be copied wholesale. `grpo/scheduler.py` therefore ports only the stochastic DDIM transition/log-probability math and reads `alpha_cumprod` from AllMerge's existing schedule.

This is the smallest change that preserves both requirements:

1. do not derive a new diffusion process;
2. do not rewrite the already-migrated AllMerge planner.

## 4. File split / line-count policy

Core responsibilities remain separated:

- `scheduler.py`: transition density only
- `sampling.py`: group generation + replay trace
- `objective.py`: advantage + ratio/clip
- `reward_adapter.py`: W4 bridge only
- `rollout_adapter.py`: environment branching only
- `checkpoint.py`: W2/GRPO checkpoint I/O only
- `trainer.py`: orchestration + optimizer only
- `train_grpo.py`: CLI only

No generated Python file may exceed 1000 lines. `apply_grpo_migration_remote.py` enforces this before copying.

## 5. Correctness gates

1. scheduler sample/replay log-prob equality;
2. same-policy `ratio_mean == 1`, `clip_fraction == 0`;
3. fake-reward 3-step smoke;
4. W2 pretrained checkpoint load with `strict=True`;
5. W4 reward adapter integration;
6. 3-step real-reward smoke;
7. 10-step smoke;
8. 100-step smoke;
9. only then restore extended ChassisFusion diagnostics / larger group sizes / Risk-PACT.

## 6. Remote source-of-truth audit

No local Diffusion-metadrive checkout is required. `apply_grpo_migration_remote.py` first resolves/downloads `KongAAAAAAJ/Diffusion-metadrive` from GitHub, scans the real remote snapshot for scheduler/log-prob, replay, GRPO objective, refinement and branch-rollout code, and writes `GRPO_SOURCE_AUDIT.md`. The audit records candidate paths, line counts and hashes so COPY/ADAPT decisions remain traceable.

The implementation rule remains migration-first: reuse mathematical GRPO/DDIM logic, adapt only the AllMerge model/environment/reward boundaries, and never copy MetaDrive/NavSim wrappers into the new trainer.
