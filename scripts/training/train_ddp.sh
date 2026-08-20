#!/bin/bash
# Multi-GPU (DDP) training launcher for dexmani_policy.
#
# Usage:
#   bash scripts/training/train_ddp.sh <config_name> [hydra_overrides...]
#
# Examples:
#   bash scripts/training/train_ddp.sh ddp/maniflow
#   bash scripts/training/train_ddp.sh ddp/dp 'training.loop.total_train_steps=100'
#   bash scripts/training/train_ddp.sh ddp/dqrise 'training.seed=123'
#
# Available DDP configs:
#   ddp/action_flow  ddp/dp  ddp/dqrise  ddp/maniflow  ddp/multitask_dit  ddp/r3d  ddp/dp3_faas  ddp/sat
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate policy

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/training/train_ddp.sh <config_name> [hydra_overrides...]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/training/train_ddp.sh ddp/maniflow"
    echo "  bash scripts/training/train_ddp.sh ddp/dp 'training.loop.total_train_steps=100'"
    echo ""
    echo "Config file is dexmani_policy/configs/<config_name>.yaml"
    exit 1
fi

CONFIG="$1"
shift

exec python dexmani_policy/train_ddp.py --config-name="${CONFIG}" "$@"
