#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/diffusion_pretrain.yaml}"
DATASET_ROOT="${DATASET_ROOT:-data/expert/allmerge_planner_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diffusion_pretrain}"

"${PYTHON_BIN}" train_diffusion_pretrain.py \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
