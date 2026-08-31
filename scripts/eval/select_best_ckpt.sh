#!/usr/bin/env bash
# Offline best-checkpoint selector via adaptive evaluation.
#
# Runs all milestone checkpoints (20/40/60/80/100%) through an adaptive
# elimination evaluation to identify the single best checkpoint.
#
# Usage:
#   bash scripts/eval/select_best_ckpt.sh <policy_name> <task_name> <exp_name> [args...]
#
# Examples:
#   bash scripts/eval/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35
#   bash scripts/eval/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35 \
#       --initial-episodes 25 --max-episodes 50
#   bash scripts/eval/select_best_ckpt.sh maniflow pick_apple_messy 2026-07-29_01-53_35 \
#       --link-best
#
# Extra args are forwarded directly to select_best_ckpt.py.
# Run with --help for the full option list.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate policy

if [[ $# -lt 3 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/eval/select_best_ckpt.sh <policy_name> <task_name> <exp_name> [args...]"
    echo ""
    echo "Positional args:"
    echo "  policy_name   policy config name (e.g. dp3, maniflow)"
    echo "  task_name     task name (e.g. pour, pick_apple_messy)"
    echo "  exp_name      experiment timestamp/name under experiments/<policy>/<task>/"
    echo ""
    echo "Extra args are forwarded to select_best_ckpt.py."
    echo ""
    echo "Options (select_best_ckpt.py):"
    echo "  --initial-episodes N   Episodes per ckpt in Phase 1 (default: 25)"
    echo "  --batch-size N         Extra episodes per round in Phase 2 (default: 5)"
    echo "  --max-episodes N       Hard cap per ckpt (default: 100)"
    echo "  --denoise-steps N      Inference denoising steps (default: from config)"
    echo "  --no-ema               Use raw weights instead of EMA"
    echo "  --seed N               Eval seed override"
    echo "  --link-best            Symlink best ckpt as checkpoints/best.pt"
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35"
    echo "  bash scripts/eval/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35 \\"
    echo "      --initial-episodes 25 --max-episodes 50"
    exit 1
fi

POLICY="$1"
TASK="$2"
EXP_NAME="$3"
shift 3

EXP_DIR="experiments/${POLICY}/${TASK}/${EXP_NAME}"

if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: experiment directory not found: ${EXP_DIR}" >&2
    echo "Check that policy_name, task_name, and exp_name are correct." >&2
    exit 1
fi

if [[ ! -f "$EXP_DIR/config.yaml" ]]; then
    echo "Error: config.yaml not found in ${EXP_DIR}" >&2
    exit 1
fi

exec python dexmani_policy/select_best_ckpt.py \
    --policy-name="${POLICY}" \
    --task-name="${TASK}" \
    --exp-name="${EXP_NAME}" \
    "$@"
