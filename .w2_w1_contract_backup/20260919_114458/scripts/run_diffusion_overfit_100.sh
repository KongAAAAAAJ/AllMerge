#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-data/expert/allmerge_planner_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diffusion_pretrain_overfit100}"

"${PYTHON_BIN}" train_diffusion_pretrain.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 100 \
  --num-workers 0 \
  --limit-train-samples 100 \
  --limit-val-samples 100 \
  --overfit-batches 1.0 \
  --max-epochs "${MAX_EPOCHS:-200}" \
  "$@"
