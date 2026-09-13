#!/bin/bash
# Multi-GPU (DDP) training launcher for dexmani_policy.
#
# Usage:
#   bash scripts/training/train_ddp.sh ddp/<config_name> [hydra_overrides...]
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

eval "$(conda shell.bash hook)"
conda activate policy

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/training/train_ddp.sh ddp/<config_name> [hydra_overrides...]"
    echo ""
    echo "Example:"
    echo "  bash scripts/training/train_ddp.sh ddp/CONFIG_NAME 'task_name=pour'"
    echo ""
    echo "DDP overlays are Hydra files under dexmani_policy/configs/ddp/."
    exit 1
fi

CONFIG="$1"
shift

exec python dexmani_policy/train_ddp.py --config-name="${CONFIG}" "$@"
