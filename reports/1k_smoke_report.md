# W2 1k-sample smoke report

Status: **runtime execution pending in the real AllMerge environment**.

Run:

```bash
DATASET_ROOT=data/expert/allmerge_planner_v1 \
  bash scripts/run_diffusion_smoke_1k.sh
```

Then evaluate and runtime-load the exported checkpoint:

```bash
CHECKPOINT=outputs/diffusion_pretrain_smoke1k/run_1/checkpoints/last.ckpt \
  NUM_SAMPLES=256 \
  bash scripts/run_diffusion_open_loop_eval.sh

python export_diffusion_runtime_checkpoint.py \
  outputs/diffusion_pretrain_smoke1k/run_1/checkpoints/last.ckpt \
  outputs/diffusion_pretrain_smoke1k/run_1/runtime/last_runtime.pt

python test_diffusion_runtime_load.py \
  --checkpoint outputs/diffusion_pretrain_smoke1k/run_1/runtime/last_runtime.pt \
  --dataset-root data/expert/allmerge_planner_v1
```

Acceptance evidence:

- no NaN/Inf or shape mismatch
- checkpoint/resume works
- open-loop `selected_ADE`, `selected_FDE`, `minADE`, `minFDE`
- raw/masked mode accuracy + top-3 raw mode hit rate
- runtime strict load PASS
- runtime output `[1,8,2]`, candidates `[1,10,8,2]`
