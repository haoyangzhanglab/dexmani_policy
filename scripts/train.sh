#!/bin/bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/train.sh <config_name> [hydra_overrides...]"
    echo "示例: bash scripts/train.sh dp3"
    exit 1
fi

CONFIG=$1
shift

python dexmani_policy/train.py --config-name="${CONFIG}" "$@"
