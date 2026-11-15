#!/usr/bin/env bash
set -euo pipefail
# Run at the AllMerge project root. For TRACE Control Diffusion (K2), not W2 XY.
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a TRACE Control Diffusion .pt}"
python -m evaluation.eval_diffusion_open_loop \
  --checkpoint "$CHECKPOINT" \
  --config "${CONFIG:-configs/train/diffusion.yaml}" \
  --split "${SPLIT:-val}" \
  --num-samples "${NUM_SAMPLES:-0}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --seeds "${SEEDS:-0,1,2,3,4}" \
  --output-dir "${OUTPUT_DIR:-outputs/evaluation/trace_open_loop}" "$@"
