# W2 Diffusion Pretraining — HANDOFF

## Status

The **code migration is complete** against the frozen AllMerge `d946cef` planner contract and the completed W1 `data_pipeline` interface.

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

W1 emits:

```python
batch = {
    "features": {...9 frozen planner feature tensors...},
    "expert_trajectory": Tensor[B, 8, 2],
    "expert_mode": Tensor[B],
    "expert_semantic": Tensor[B],
    "metadata": list[dict],
}
```

W2 performs exactly:

```python
output = planner.forward_train(
    batch["features"],
    batch["expert_trajectory"],
    batch["expert_semantic"],
)
loss = output["loss"]
```

`expert_mode` is compared to `output["target_mode"]` as a diagnostic only (`w1_target_mode_match`). It never becomes a second source of training-label logic.

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

## Formal training

```bash
DATASET_ROOT=data/expert/allmerge_planner_v1 \
OUTPUT_DIR=outputs/diffusion_pretrain \
  bash scripts/run_diffusion_pretrain.sh
```

Resume:

```bash
DATASET_ROOT=data/expert/allmerge_planner_v1 \
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
  --dataset-root data/expert/allmerge_planner_v1
```
