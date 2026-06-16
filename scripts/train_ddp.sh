#!/bin/bash

set -euo pipefail

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "用法: bash scripts/train_ddp.sh <config_name> [hydra_overrides...]" >&2
    echo "示例: bash scripts/train_ddp.sh ddp/maniflow" >&2
    echo "      bash scripts/train_ddp.sh ddp/dp 'training.loop.num_epochs=100'" >&2
    exit 1
fi

CONFIG=$1
shift

CONFIG_FILE="dexmani_policy/configs/${CONFIG}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE" >&2
    exit 1
fi

# GPU 可用性预检查：比 Hydra/DDP 报错更友好
NUM_GPUS=$(grep -oP '^\s+num_gpus:\s*\K\d+' "$CONFIG_FILE" 2>/dev/null || echo "0")
if [ "${NUM_GPUS:-0}" -gt 0 ]; then
    AVAIL=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [ "$NUM_GPUS" -gt "$AVAIL" ]; then
        echo "错误: 配置要求 $NUM_GPUS GPUs，但仅 $AVAIL 可用" >&2
        exit 1
    fi
fi

python dexmani_policy/train_ddp.py --config-name="${CONFIG}" "$@"
