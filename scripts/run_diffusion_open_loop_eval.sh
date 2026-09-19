#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a W2 .ckpt or exported runtime .pt}"
DATASET_ROOT="${DATASET_ROOT:-data/expert/allmerge_planner_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diffusion_pretrain/open_loop_eval}"

"${PYTHON_BIN}" eval_diffusion_open_loop.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split val \
  --num-samples "${NUM_SAMPLES:-1000}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
