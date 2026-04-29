#!/bin/bash
# 单卡训练
# 用法: bash scripts/train.sh <config_name> [overrides...]
# 示例: bash scripts/train.sh dp3 task_name=sim/pick_apple_messy seed=42

set -euo pipefail

CONFIG=${1:?"用法: bash scripts/train.sh <config_name> [overrides...]"}
shift

python dexmani_policy/train.py --config-name="${CONFIG}" "$@"
