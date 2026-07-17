#!/bin/bash
# VQ-VAE hand-pose pretraining launcher.
#
# Usage:
#   bash scripts/train_vq_hand.sh [task_name] [cli_overrides...]
#
# Examples:
#   bash scripts/train_vq_hand.sh                           # default: pick_apple_messy
#   bash scripts/train_vq_hand.sh pour                      # task=pour
#   bash scripts/train_vq_hand.sh pick_apple_messy --num_epochs 2000 --lr 1e-4
#   TASK_NAME=pour bash scripts/train_vq_hand.sh             # env-var override
#   ZARR_PATH=/custom/data.zarr bash scripts/train_vq_hand.sh
#
# See the vq_vae section in configs/dqrise.yaml for all configurable fields.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ── Task name (first positional arg, before any --flags) ─────────────────────
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    TASK_NAME="$1"
    shift
else
    TASK_NAME="${TASK_NAME:-pick_apple_messy}"
fi

ZARR_PATH="${ZARR_PATH:-robot_data/${TASK_NAME}.zarr}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/vq_hand/${TASK_NAME}}"

exec python -u -m dexmani_policy.scripts.train_vq_hand \
    --config dexmani_policy/configs/dqrise.yaml \
    --zarr_path "${ZARR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
