#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/train/diffusion.yaml}"
DATASET_ROOT="${DATASET_ROOT:-outputs/expert_dataset/expert_trucksim_k2_anchor_pair_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diffusion_pretrain_smoke1k}"
"${PYTHON_BIN}" scripts/training/run_diffusion.py \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit-samples 1000 \
  --max-epochs 1 \
  --max-steps "${MAX_STEPS:-100}" \
  "$@"
