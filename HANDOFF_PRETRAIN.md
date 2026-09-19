# W2 Diffusion Pretraining — HANDOFF

## Status

The **code migration is complete** against the frozen AllMerge `d946cef` planner contract and the completed W1 root-level `expert_dataset.py` interface.

The migration preserves the proven Diffusion-metadrive training skeleton while replacing only the dataset/model interfaces. It does not rewrite the planner and does not move any target/loss logic into the Trainer.

Real 100-sample overfit and 1k smoke training are **not claimed as executed in this build sandbox** because the real W1 expert shards and full AllMerge runtime are not mounted here. Scripts and report targets are included so those two acceptance runs can be executed directly in the actual project environment.

## Added files

```text
PRETRAIN_PORTING_MAP.md
train_diffusion_pretrain.py
eval_diffusion_open_loop.py
export_diffusion_runtime_checkpoint.py
test_diffusion_runtime_load.py

pretraining/
  __init__.py
  checkpoint_io.py
  contract.py
  dataset_adapter.py
  lightning_module.py
  warmup_cos_lr.py
  smoke_test_contract.py

configs/
  diffusion_pretrain.yaml

scripts/
  run_diffusion_pretrain.sh
  run_diffusion_open_loop_eval.sh
  run_diffusion_overfit_100.sh
  run_diffusion_smoke_1k.sh

reports/
  100_sample_overfit_report.md
  1k_smoke_report.md
```

## W1 -> W2 exact adapter

W1 `expert_dataset.py` returns `(features, targets)` (or `(features, targets, metadata)` when requested):

```python
features = {...9 frozen planner feature tensors...}
targets = {
    "trajectory": Tensor[B, 8, 2],
    "target_mode": Tensor[B],
    "target_semantic": Tensor[B],
}
```

`pretraining/dataset_adapter.py` only renames this existing contract for W2 orchestration. The model call remains:

```python
output = planner.forward_train(
    features,
    targets["trajectory"],
    targets["target_semantic"],
)
loss = output["loss"]
```

`targets["target_mode"]` is compared to `output["target_mode"]` as a diagnostic only (`w1_target_mode_match`). It never becomes a second source of training-label logic.

## Training structure retained from Diffusion-metadrive

```text
ExpertDataset
-> DataLoader
-> DiffusionPretrainModule
-> StructuredDiffusionPlanner.forward_train()
-> model-owned loss
-> AdamW
-> WarmupCosLR
-> TensorBoard
-> ModelCheckpoint
-> validation
```

## Checkpoint / resume contract

Two paths are intentionally separated:

1. `--resume-from-checkpoint`: Lightning resume, restoring model, optimizer, scheduler and epoch/step.
2. `--init-checkpoint`: planner weights only, for warm-starting a fresh training run.

Every Lightning checkpoint keeps the normal Lightning `state_dict` and additionally stores:

```text
planner_state_dict
allmerge_model_config
checkpoint_format=allmerge_diffusion_pretrain_v1
```

`export_diffusion_runtime_checkpoint.py` converts the best/last `.ckpt` to a planner-only runtime `.pt` without modifying the frozen `DiffusionPlannerRuntime` implementation.

## Open-loop metrics

`eval_diffusion_open_loop.py` reports:

- selected ADE / FDE
- minADE / minFDE across 10 candidates
- raw mode accuracy
- traffic-masked mode accuracy
- top-3 raw mode hit rate
- expert-mode traffic-valid fraction

Raw and masked mode metrics are kept separate because AllMerge training assignment intentionally does not use online `mode_valid_mask`, while runtime selection does.

## First commands in the real AllMerge checkout

```bash
python -m py_compile \
  train_diffusion_pretrain.py \
  eval_diffusion_open_loop.py \
  export_diffusion_runtime_checkpoint.py \
  test_diffusion_runtime_load.py \
  pretraining/*.py

python -m pretraining.smoke_test_contract
```

Then W1 real-data validation, W2 100-sample overfit, W2 1k smoke, and finally full pretraining.


## Current 100-sample overfit dataset

The completed W1 100-sample set contains one shard. Its shard-level split therefore has no independent validation shard. For the overfit gate, intentionally use the same fixed 100 samples for train and validation:

```bash
python train_diffusion_pretrain.py \
  --dataset-root outputs/expert_dataset/allmerge_overfit100_2 \
  --train-split all \
  --val-split all \
  --batch-size 100 \
  --num-workers 0 \
  --limit-train-samples 100 \
  --limit-val-samples 100 \
  --overfit-batches 1.0 \
  --max-epochs 200
```

## Formal training

```bash
DATASET_ROOT=outputs/expert_dataset/allmerge_expert \
OUTPUT_DIR=outputs/diffusion_pretrain \
  bash scripts/run_diffusion_pretrain.sh
```

Resume:

```bash
DATASET_ROOT=outputs/expert_dataset/allmerge_expert \
OUTPUT_DIR=outputs/diffusion_pretrain \
  bash scripts/run_diffusion_pretrain.sh \
  --resume-from-checkpoint outputs/diffusion_pretrain/run_1/checkpoints/last.ckpt
```

Open-loop evaluation:

```bash
CHECKPOINT=outputs/diffusion_pretrain/run_1/runtime/best_runtime.pt \
  bash scripts/run_diffusion_open_loop_eval.sh
```

Runtime load test:

```bash
python test_diffusion_runtime_load.py \
  --checkpoint outputs/diffusion_pretrain/run_1/runtime/best_runtime.pt \
  --dataset-root outputs/expert_dataset/allmerge_expert
```
