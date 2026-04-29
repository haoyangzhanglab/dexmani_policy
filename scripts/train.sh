#!/bin/bash
# 单卡训练
# 用法: bash scripts/train.sh <config_name> [hydra_overrides...]

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/train.sh <config_name> [hydra_overrides...]"
    echo ""
    echo "示例:"
    echo "  bash scripts/train.sh dp3"
    echo "  bash scripts/train.sh maniflow"
    echo ""
    echo "  # 指定 GPU（多卡服务器）"
    echo "  bash scripts/train.sh dp3 training.device=cuda:2"
    echo "  CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh dp3"
    echo ""
    echo "  # 覆盖配置"
    echo "  bash scripts/train.sh dp3 task_name=sim/pick_apple_messy training.seed=42"
    exit 1
fi

CONFIG=$1
shift

python dexmani_policy/train.py --config-name="${CONFIG}" "$@"
