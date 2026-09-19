# HANDOFF_PRETRAIN

## Status

W2 diffusion pretraining is connected to the accepted W1 expert dataset and the
frozen `StructuredDiffusionPlanner.forward_train()` API.

The training backend intentionally uses **pure PyTorch**, not PyTorch Lightning,
so it runs in the existing AllMerge requirements environment without adding a
new training framework. Configuration defaults to
`configs/diffusion_pretrain.json`, avoiding a mandatory PyYAML dependency.

## Main files

```text
train_diffusion_pretrain.py
pretraining/
  config_io.py
  dataset_adapter.py
  contract.py
  trainer.py
  warmup_cos_lr.py
  checkpoint_io.py
configs/
  diffusion_pretrain.json
eval_diffusion_open_loop.py
export_diffusion_runtime_checkpoint.py
test_diffusion_runtime_load.py
```

All formal Python files remain below 1000 lines.

## W1 -> W2 contract

W1 returns `(features, targets)` where `targets` contains `trajectory`,
`target_mode`, and `target_semantic`. W2 forwards only the structured features,
trajectory, and semantic label to `forward_train()`; W1 `target_mode` is used
only for the `w1_target_mode_match` diagnostic.

## 100-sample overfit

For the accepted one-shard 100-sample dataset, train and validation both use
`split=all`; this is a memorization/correctness gate, not a generalization test.

```bat
python train_diffusion_pretrain.py ^
  --dataset-root outputs\expert_dataset\allmerge_overfit100_2 ^
  --train-split all ^
  --val-split all ^
  --batch-size 100 ^
  --num-workers 0 ^
  --limit-train-samples 100 ^
  --limit-val-samples 100 ^
  --overfit-batches 1.0 ^
  --max-epochs 200
```

Acceptance focus: loss decreases strongly, no NaN/Inf, checkpoint is written,
and `w1_target_mode_match` remains near 1.0.

## 1k smoke

After overfit passes, collect/validate W1 1k data and run the 1k smoke script or
the equivalent CLI. Only after that should the runtime-load test and formal
pretraining be treated as ready.

## Checkpoint format

Training checkpoint contains planner weights, optimizer, scheduler, AMP scaler,
epoch/global step, model config and resolved train config. Runtime export writes
a planner-only `state_dict` consumable by the frozen `DiffusionPlannerRuntime`.
