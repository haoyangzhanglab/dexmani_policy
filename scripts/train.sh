#!/bin/bash

set -euo pipefail

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "用法: bash scripts/train.sh <config_name> [hydra_overrides...]" >&2
    echo "示例: bash scripts/train.sh dp3" >&2
    echo "      bash scripts/train.sh multitask_dit" >&2
    echo "      bash scripts/train.sh dp3 'training.loop.num_epochs=10'" >&2
    exit 1
fi

CONFIG=$1
shift

CONFIG_FILE="dexmani_policy/configs/${CONFIG}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE" >&2
    exit 1
fi

python dexmani_policy/train.py --config-name="${CONFIG}" "$@"
