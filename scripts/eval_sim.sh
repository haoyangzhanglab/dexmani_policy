#!/bin/bash
# 仿真评估
# 用法: bash scripts/eval_sim.sh <policy_name> <task_name> <exp_name> [hydra_overrides...]

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "用法: bash scripts/eval_sim.sh <policy_name> <task_name> <exp_name> [hydra_overrides...]"
    echo ""
    echo "示例:"
    echo "  bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233"
    echo ""
    echo "  # 测试不同的 denoise_timesteps"
    echo "  bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233 eval.sim.denoise_timesteps_list='[5,10,20]'"
    echo ""
    echo "  # 指定 checkpoint 和评估 episode 数"
    echo "  bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233 eval.sim.ckpt_tag_or_path=best eval.sim.eval_episodes=200"
    echo ""
    echo "  # 使用非 EMA 模型"
    echo "  bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233 eval.sim.use_ema_for_eval=False"
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
