#!/bin/bash
# Multi-GPU (DDP) training launcher for dexmani_policy.
#
# Usage:
#   bash scripts/training/train_ddp.sh ddp/<config_name> [hydra_overrides...]
#
set -euo pipefail

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/training/train_ddp.sh ddp/<config_name> [hydra_overrides...]"
    echo ""
    echo "Example:"
    echo "  bash scripts/training/train_ddp.sh ddp/CONFIG_NAME 'task_name=pour'"
    echo ""
    echo "DDP overlays are Hydra files under dexmani_policy/configs/ddp/."
    if [[ $# -eq 0 ]]; then
        exit 1
    fi
    exit 0
fi

CONFIG="$1"
shift

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# Keep Conda activation hooks outside this shell's nounset mode and stream logs.
exec conda run --no-capture-output -n policy \
    python -u dexmani_policy/train_ddp.py --config-name="${CONFIG}" "$@"
