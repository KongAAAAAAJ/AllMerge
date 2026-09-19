#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-outputs/expert_dataset/allmerge_smoke1k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diffusion_pretrain_smoke1k}"

"${PYTHON_BIN}" train_diffusion_pretrain.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --train-split train \
  --val-split test \
  --limit-train-samples 1000 \
  --limit-val-samples 256 \
  --max-steps "${MAX_STEPS:-100}" \
  "$@"
