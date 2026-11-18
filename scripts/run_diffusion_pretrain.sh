#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/train/diffusion.yaml}"
DATASET_ROOT="${DATASET_ROOT:-outputs/expert_dataset/expert_trucksim_k2_raw_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/final_compare/control_v0}"
"${PYTHON_BIN}" scripts/training/run_diffusion.py \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
