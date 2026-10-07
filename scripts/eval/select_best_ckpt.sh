#!/usr/bin/env bash
# Select among milestones using fixed initial seeds and an optional exact-tie batch.
# Usage: bash scripts/eval/select_best_ckpt.sh <policy> <task> <exp> [args...]
# Extra arguments, including --help, go to select_best_ckpt.py.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
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
    echo "  --max-episodes N       Hard cap for selection + reserved tie-break seeds (default: 100)"
    echo "  --inference-steps N      Inference steps (default: from config)"
    echo "  --no-ema               Use raw weights instead of EMA"
    echo "  --seed N               Eval seed override"
    echo "  --videos               Record isolated candidate/stage videos (default: off)"
    echo "  --result-file PATH     Write this selection to a new handoff file"
    echo "  eval.seed_manifest=PATH  Required fixed roles (or saved config); must fit --max-episodes"
    echo ""
    echo "  Dot-list overrides may change eval/environment controls; model inputs come from saved config.yaml and cannot be overridden."
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35"
    echo "  bash scripts/eval/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35 \\"
    echo "      eval.seed_manifest=/absolute/path/seeds.json --max-episodes 50"
    exit 0
fi

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/eval/select_best_ckpt.sh <policy_name> <task_name> <exp_name> [args...]" >&2
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

exec conda run --no-capture-output -n policy python dexmani_policy/select_best_ckpt.py \
    --policy-name="${POLICY}" \
    --task-name="${TASK}" \
    --exp-name="${EXP_NAME}" \
    "$@"
