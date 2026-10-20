#!/bin/bash
# Policy-aligned DQ codebook training.
# Usage: vqvae.sh [task_name] [VQ training options...]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

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
    python -u -m dexmani_policy.training.pretrain.hand_prior.vqvae \
    --policy-config dexmani_policy/configs/dqrise.yaml \
    --policy-override "task_name=${TASK_NAME}" \
    --policy-override "dataset.zarr_path=${ZARR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
