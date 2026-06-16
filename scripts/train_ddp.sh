#!/bin/bash
# Multi-GPU (DDP) training launcher for dexmani_policy.
#
# Usage:
#   bash scripts/train_ddp.sh <config_name> [hydra_overrides...]
#
# Examples:
#   bash scripts/train_ddp.sh ddp/maniflow
#   bash scripts/train_ddp.sh ddp/dp 'training.loop.num_epochs=100'
#   bash scripts/train_ddp.sh ddp/multitask_dit 'training.seed=123'
#
# Available DDP configs:
#   ddp/dp  ddp/maniflow  ddp/multitask_dit  ddp/r3d
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/train_ddp.sh <config_name> [hydra_overrides...]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/train_ddp.sh ddp/maniflow"
    echo "  bash scripts/train_ddp.sh ddp/dp 'training.loop.num_epochs=100'"
    echo ""
    echo "Config file is dexmani_policy/configs/<config_name>.yaml"
    exit 1
fi

CONFIG="$1"
shift

# GPU availability pre-check: friendlier error than downstream DDP failure.
# Uses OmegaConf to properly resolve config inheritance (ddp configs
# compose on top of base configs via Hydra defaults).
AVAIL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
CONFIG_GPUS=$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('dexmani_policy/configs/${CONFIG}.yaml')
ng = cfg.get('training', {}).get('num_gpus', 0)
print(ng if isinstance(ng, int) else 0)
" 2>/dev/null || echo "0")

if [[ "${CONFIG_GPUS:-0}" -gt "${AVAIL_GPUS:-0}" ]]; then
    echo "Error: config requires ${CONFIG_GPUS} GPUs, but only ${AVAIL_GPUS} available." >&2
    exit 1
fi

exec python dexmani_policy/train_ddp.py --config-name="${CONFIG}" "$@"
