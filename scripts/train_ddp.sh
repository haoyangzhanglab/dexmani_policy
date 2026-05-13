#!/bin/bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/train_ddp.sh <config_name> [hydra_overrides...]"
    echo "示例: bash scripts/train_ddp.sh maniflow_ddp"
    exit 1
fi

CONFIG=$1
shift

python dexmani_policy/train_ddp.py --config-name="${CONFIG}" "$@"
