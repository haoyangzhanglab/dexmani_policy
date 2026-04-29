#!/bin/bash
# DDP 多卡训练（必须使用 train_ddp.py，config 须含 training.num_gpus）
# 用法: bash scripts/train_ddp.sh <config_name> [overrides...]
# 示例: bash scripts/train_ddp.sh maniflow_ddp task_name=sim/pick_apple_messy
# 示例（指定 GPU）: bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[0,1,2,3]
# 示例（环境变量选卡）: CUDA_VISIBLE_DEVICES=2,3,5,6 bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=4 training.gpu_ids=null

set -euo pipefail

CONFIG=${1:?"用法: bash scripts/train_ddp.sh <config_name> [overrides...]"}
shift

python dexmani_policy/train_ddp.py --config-name="${CONFIG}" "$@"
