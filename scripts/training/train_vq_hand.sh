#!/bin/bash
# Policy-aligned DQ codebook training.
# Usage: train_vq_hand.sh [task_name] [VQ training options...]
# Independent VQ research remains explicitly available via the Python --config CLI.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ── Task name (first positional arg, before any --flags) ─────────────────────
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    TASK_NAME="$1"
    shift
else
    TASK_NAME="${TASK_NAME:-pick_apple_messy}"
fi

ZARR_PATH="${ZARR_PATH:-robot_data/${TASK_NAME}.zarr}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/vq_hand/${TASK_NAME}/$(date +%Y%m%d_%H%M%S)_${RANDOM}}"

# Keep Conda activation hooks outside this shell's nounset mode and stream logs.
exec conda run --no-capture-output -n policy \
    python -u scripts/training/train_vq_hand.py \
    --policy-config dexmani_policy/configs/dqrise.yaml \
    --policy-override "task_name=${TASK_NAME}" \
    --policy-override "dataset.zarr_path=${ZARR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
