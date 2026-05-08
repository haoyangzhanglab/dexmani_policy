#!/bin/bash
# 多任务单卡训练
# 用法: bash scripts/train_multi_task.sh <config_name> [hydra_overrides...]

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/train_multi_task.sh <config_name> [hydra_overrides...]"
    echo ""
    echo "示例:"
    echo "  bash scripts/train_multi_task.sh multi_task_dp3"
    echo ""
    echo "  # 指定 GPU（多卡服务器）"
    echo "  bash scripts/train_multi_task.sh multi_task_dp3 training.device=cuda:2"
    echo "  CUDA_VISIBLE_DEVICES=3 bash scripts/train_multi_task.sh multi_task_dp3"
    echo ""
    echo "  # 覆盖配置"
    echo "  bash scripts/train_multi_task.sh multi_task_dp3 training.seed=123"
    exit 1
fi

CONFIG=$1
shift

python dexmani_policy/train_multi_task.py --config-name="${CONFIG}" "$@"
