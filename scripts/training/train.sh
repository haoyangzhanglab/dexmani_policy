#!/bin/bash
# Single-GPU training launcher for dexmani_policy.
#
# Usage:
#   bash scripts/training/train.sh <config_name> [hydra_overrides...]
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

eval "$(conda shell.bash hook)"
conda activate policy

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/training/train.sh <config_name> [hydra_overrides...]"
    echo ""
    echo "Example:"
    echo "  bash scripts/training/train.sh CONFIG_NAME 'task_name=pour' 'training.seed=42'"
    echo ""
    echo "Configs are Hydra files under dexmani_policy/configs/."
    exit 1
fi

CONFIG="$1"
shift

exec python dexmani_policy/train.py --config-name="${CONFIG}" "$@"
