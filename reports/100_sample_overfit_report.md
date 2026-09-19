# W2 100-sample overfit report

Status: **runtime execution pending in the real AllMerge environment**.

Reason: the build sandbox used to assemble this migration does not contain the user's real W1 expert shards or the full AllMerge simulator checkout. No synthetic result is presented as a real acceptance result.

Run:

```bash
DATASET_ROOT=data/expert/allmerge_planner_v1 \
  bash scripts/run_diffusion_overfit_100.sh
```

Acceptance evidence to paste here from `train_result.json` / TensorBoard:

- train/loss initial -> final
- train/trajectory_regression_loss initial -> final
- train/trajectory_classification_loss initial -> final
- val/loss
- train/w1_target_mode_match (expected 1.0)
- best Lightning checkpoint
- exported `runtime/best_runtime.pt`

Gate: a tiny fixed dataset should show a clear loss decrease; `w1_target_mode_match` must remain 1.0.
