#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/diffusion_pretrain.json}"
DATASET_ROOT="${DATASET_ROOT:-outputs/expert_dataset/allmerge_expert}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diffusion_pretrain}"

"${PYTHON_BIN}" train_diffusion_pretrain.py \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
