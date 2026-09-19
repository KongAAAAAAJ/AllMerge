# HANDOFF_GRPO

## What this migration changes

It adds a thin GRPO package around the existing `StructuredDiffusionPlanner`. It does not change the frozen planner trunk and does not duplicate W4 reward logic.

New files:

```text
highway_env/planner/diffusion/grpo/
  __init__.py
  scheduler.py
  sampling.py
  objective.py
  reward_adapter.py
  rollout_adapter.py
  checkpoint.py
  trainer.py
train_grpo.py
configs/train/allmerge_grpo.yaml
tests/grpo/*
GRPO_PORTING_MAP.md
HANDOFF_GRPO.md
```

## Apply

From the extracted migration bundle. The installer downloads Diffusion-metadrive directly from GitHub; no local source checkout is needed:

```bat
python apply_grpo_migration_remote.py ^
  --target-root D:\KONG_Files\AllMerge\all_merge ^
  --run-tests
```

It verifies that the target contains AllMerge baseline `63a7365`, downloads the current remote `Diffusion-metadrive/main`, writes `GRPO_SOURCE_AUDIT.md`, and only then installs W3. Use `--dry-run` first if you only want the remote source audit.

Use `--force` only when intentionally replacing a previous W3 migration. Existing files are backed up under `.grpo_migration_backup/` first.

## Gate 0: static / unit validation

```bash
python -m pytest -q tests/grpo
```

Expected critical assertions:

- replayed DDIM log-prob equals sampled log-prob;
- same-policy ratio is 1;
- clip fraction is 0 before an update;
- fake-reward GRPO updates trainable trajectory-head parameters;
- rollout branch restores/forks state without trainer-specific simulator code.

## Gate 1: W2 checkpoint + fake reward

Use the actual W1 shard and W2 pretrained checkpoint:

```bash
python train_grpo.py \
  --checkpoint <W2_PRETRAINED.pt> \
  --dataset-shard outputs/expert_dataset/<DATASET>/shards/shard_000000.npz \
  --steps 3 \
  --group-size 4 \
  --fake-reward
```

Then repeat with `--steps 10` and `--steps 100` only after the previous gate is clean.

## Gate 2: W4 reward

If W4 exposes one of the auto-detected paths, omit `--reward-fn`. Otherwise pass its real symbol without copying reward math:

```bash
python train_grpo.py \
  --checkpoint <W2_PRETRAINED.pt> \
  --dataset-shard <W1_SHARD.npz> \
  --steps 3 \
  --reward-fn your.reward.module:evaluate_candidates
```

The adapter expects one scalar reward per trajectory candidate. If W4's final function uses a different call signature, modify only `grpo/reward_adapter.py`; do not move reward code into `trainer.py`.

## Gate 3: live scenario rollout

`AllMergeRolloutAdapter` deliberately prefers native `get_state/set_state` when the environment gains them. Until then it uses `deepcopy` branches for correctness. Keep this slow fallback for the first correctness runs. Optimize snapshotting only after 100-step GRPO behavior is verified.

## Training semantics

Default trainable modules:

```text
denoiser.layers
denoiser.reg_head
```

Frozen by default:

```text
scene_encoder
denoiser.anchor_encoder
denoiser.mode_embedding
denoiser.time_embedding
denoiser.cls_head
```

This keeps the migration close to the old `diff_decoder`-only refinement intent while avoiding changes to the AllMerge architecture.

## Metrics to watch first

For the first 3/10/100-step runs, prioritize:

```text
reward_mean
reward_std
ratio_mean
clip_fraction
approx_kl
reference_kl
grad_norm
advantage_mean
advantage_std
```

Only after these are sane should the existing ChassisFusion diagnostics (`vehicle_reward_gain`, `paired_n48_reward_gain`, `selected_vehicle_reward_gain`) be restored into the AllMerge trainer.

## Stop conditions for debugging

Stop the smoke run and inspect before scaling up when any of these occurs:

- non-finite loss/log-prob/gradient;
- same-policy ratio deviates materially from 1 before optimization;
- reward standard deviation is effectively zero for many groups;
- `clip_fraction` is immediately saturated;
- `reference_kl` jumps sharply on the first few updates;
- W2 checkpoint does not load strictly;
- trainable parameters include the scene encoder or classifier unexpectedly.
