#!/bin/bash

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "用法: bash scripts/eval_sim.sh <policy_name> <task_name> <exp_name> [hydra_overrides...]"
    echo "示例: bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233"
    exit 1
fi

POLICY=$1
TASK=$2
EXP=$3
shift 3

python dexmani_policy/eval_sim.py \
    --policy-name="${POLICY}" \
    --task-name="${TASK}" \
    --exp-name="${EXP}" \
    "$@"
