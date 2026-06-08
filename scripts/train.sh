#!/bin/bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/train.sh <config_name> [hydra_overrides...]" >&2
    echo "示例: bash scripts/train.sh dp3" >&2
    echo "      bash scripts/train.sh multitask_dit" >&2
    echo "      bash scripts/train.sh dp3 'training.loop.num_epochs=10'" >&2
    exit 1
fi

CONFIG=$1
shift

python dexmani_policy/train.py --config-name="${CONFIG}" "$@"
