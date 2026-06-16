#!/bin/bash
# Simulation evaluation launcher for dexmani_policy.
#
# Usage:
#   bash scripts/eval_sim.sh <policy_name> <task_name> <exp_name> [overrides...]
#
# Examples:
#   bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233
#   bash scripts/eval_sim.sh maniflow pick_apple_messy 2026-04-01_11-18_233 \
#       eval.offline.denoise_timesteps_list=[5,10,20]
#
# Overrides use OmegaConf dot-list format and are merged onto the
# experiment's saved config.yaml.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 3 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/eval_sim.sh <policy_name> <task_name> <exp_name> [overrides...]"
    echo ""
    echo "Positional args:"
    echo "  policy_name   policy config name (e.g. dp3, maniflow)"
    echo "  task_name     task name (e.g. pick_apple_messy)"
    echo "  exp_name      experiment timestamp/name under experiments/<policy>/<task>/"
    echo ""
    echo "Overrides use OmegaConf dot-list format, merged onto config.yaml."
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233"
    echo "  bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233 \\"
    echo "      eval.offline.denoise_timesteps_list=[5,10,20]"
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

exec python dexmani_policy/eval_sim.py \
    --policy-name="${POLICY}" \
    --task-name="${TASK}" \
    --exp-name="${EXP_NAME}" \
    "$@"
