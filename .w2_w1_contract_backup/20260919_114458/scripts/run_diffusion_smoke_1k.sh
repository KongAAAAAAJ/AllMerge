#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-data/expert/allmerge_planner_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diffusion_pretrain_smoke1k}"

"${PYTHON_BIN}" train_diffusion_pretrain.py \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit-train-samples 1000 \
  --limit-val-samples 256 \
  --max-steps "${MAX_STEPS:-100}" \
  "$@"
