#!/bin/bash
# DDP 多卡训练
# 用法: bash scripts/train_ddp.sh <config_name> [hydra_overrides...]

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/train_ddp.sh <config_name> [hydra_overrides...]"
    echo ""
    echo "示例:"
    echo "  # 使用默认 GPU (0,1,2,3)"
    echo "  bash scripts/train_ddp.sh maniflow_ddp"
    echo ""
    echo "  # 指定 GPU"
    echo "  bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[0,1,2,3]"
    echo "  bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[1,2,3,4]"
    echo ""
    echo "  # 使用非连续 GPU"
    echo "  bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=2 training.gpu_ids=[0,7]"
    echo ""
    echo "  # 通过环境变量选卡"
    echo "  CUDA_VISIBLE_DEVICES=2,3,5,6 bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=4"
    exit 1
fi

CONFIG=$1
shift

python dexmani_policy/train_ddp.py --config-name="${CONFIG}" "$@"
