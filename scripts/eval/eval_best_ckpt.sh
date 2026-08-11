#!/usr/bin/env bash
# RoboTwin-style checkpoint evaluation — simple success rate.
#
# Loads a checkpoint and evaluates it on all seeds from the pool.
# Output is the success rate (matching RoboTwin _result.txt format).
#
# Usage:
#   bash scripts/eval/eval_best_ckpt.sh <policy_name> <task_name> <exp_name> [args...]
#
# Examples:
#   bash scripts/eval/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42
#   bash scripts/eval/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42 --ckpt-tag 20pct
#   bash scripts/eval/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42 --episodes 50
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate policy

if [[ $# -lt 3 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/eval/eval_best_ckpt.sh <policy_name> <task_name> <exp_name> [args...]"
    echo ""
    echo "Positional args:"
    echo "  policy_name   policy config name (e.g. dp3, maniflow)"
    echo "  task_name     task name (e.g. pour, pick_apple_messy)"
    echo "  exp_name      experiment timestamp/name under experiments/<policy>/<task>/"
    echo ""
    echo "Options (eval_best_ckpt.py):"
    echo "  --ckpt-tag TAG       Checkpoint: best, latest, 20pct..100pct (default: best)"
    echo "                         'best' reads best_ckpt.json written by select_best_ckpt.sh"
    echo "  --ckpt-path PATH     Direct .pt path (overrides --ckpt-tag)"
    echo "  --episodes N         Number of seeds (default: 100)"
    echo "  --denoise-steps N    Single inference step count (default: from config)"
    echo "                       To sweep multiple denoise steps, set in config:"
    echo "                         eval.denoise_timesteps_list=[5,10,20]"
    echo "  --no-ema             Use raw weights instead of EMA"
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42"
    echo "  bash scripts/eval/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42 --ckpt-tag 40pct"
    echo "  bash scripts/eval/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42 --episodes 50"
    exit 1
fi

POLICY="$1"
TASK="$2"
EXP_NAME="$3"
shift 3

EXP_DIR="experiments/${POLICY}/${TASK}/${EXP_NAME}"

if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: experiment directory not found: ${EXP_DIR}" >&2
    exit 1
fi

if [[ ! -f "$EXP_DIR/config.yaml" ]]; then
    echo "Error: config.yaml not found in ${EXP_DIR}" >&2
    exit 1
fi

# Validate at least one checkpoint exists
if [[ ! -d "$EXP_DIR/checkpoints" ]] || \
   ! compgen -G "$EXP_DIR/checkpoints/epoch=*.pt" > /dev/null; then
    echo "Error: no checkpoints found in ${EXP_DIR}/checkpoints/" >&2
    echo "The experiment may not have completed any training steps." >&2
    exit 1
fi

exec python dexmani_policy/eval_best_ckpt.py \
    --policy-name="${POLICY}" \
    --task-name="${TASK}" \
    --exp-name="${EXP_NAME}" \
    "$@"
