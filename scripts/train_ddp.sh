#!/bin/bash
# DDP 多卡训练启动脚本
# 用法: bash scripts/train_ddp.sh <config_name> <task_name> <num_gpus>
# 示例: bash scripts/train_ddp.sh dp3 sim/pick_apple_messy 4

CONFIG=$1
TASK=$2
NUM_GPUS=${3:-4}

if [ -z "$CONFIG" ] || [ -z "$TASK" ]; then
    echo "用法: bash scripts/train_ddp.sh <config_name> <task_name> [num_gpus]"
    echo "示例: bash scripts/train_ddp.sh dp3 sim/pick_apple_messy 4"
    exit 1
fi

echo "=========================================="
echo "DDP 多卡训练"
echo "配置: ${CONFIG}"
echo "任务: ${TASK}"
echo "GPU 数量: ${NUM_GPUS}"
echo "=========================================="

python dexmani_policy/train.py \
    --config-name=${CONFIG}.yaml \
    task_name=${TASK} \
    training.num_gpus=${NUM_GPUS}
