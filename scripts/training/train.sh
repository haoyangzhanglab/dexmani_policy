#!/bin/bash
# Single-GPU training launcher for dexmani_policy.
#
# Usage:
#   bash scripts/training/train.sh <config_name> [hydra_overrides...]
#
# Examples:
#   bash scripts/training/train.sh dp3
#   bash scripts/training/train.sh dp3 'training.loop.total_train_steps=10'
#   bash scripts/training/train.sh maniflow 'training.seed=42'
#   bash scripts/training/train.sh multitask_dit
#
# Available configs (top-level, non-ddp):
#   dp  dp3  maniflow  moe_dp  multitask_dit  r3d  dqrise  dp3_faas
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/training/train.sh <config_name> [hydra_overrides...]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/training/train.sh dp3"
    echo "  bash scripts/training/train.sh multitask_dit"
    echo "  bash scripts/training/train.sh dp3 'training.loop.total_train_steps=10'"
    echo ""
    echo "Config file is dexmani_policy/configs/<config_name>.yaml"
    exit 1
fi

CONFIG="$1"
shift

exec python dexmani_policy/train.py --config-name="${CONFIG}" "$@"
