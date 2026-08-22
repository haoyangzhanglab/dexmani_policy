#!/usr/bin/env bash
# Paired ActionFlow solver ablation on one checkpoint and one deterministic seed set.
#
# Usage:
#   bash scripts/eval/eval_action_flow_solvers.sh \
#       <policy> <task> <exp_name> [--episodes N] [--ckpt-tag TAG] [--no-ema]
#
# The fixed screening candidates are Euler-2, Midpoint-2, Euler-4,
# Midpoint-4, and Midpoint-8. Euler itself accepts any positive NFE (including
# 10), but Euler-10 is intentionally not part of this first-pass sweep.
# Re-run selected candidates with --episodes 100 before changing the default.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/eval/eval_action_flow_solvers.sh <policy> <task> <exp_name> [options]"
    echo ""
    echo "Options:"
    echo "  --episodes N    Paired episode count (default: 25)"
    echo "  --ckpt-tag TAG  best, latest, or 20pct..100pct (default: best)"
    echo "  --no-ema        Evaluate raw rather than EMA weights"
    exit 0
fi

if [[ $# -lt 3 ]]; then
    echo "Error: policy, task, and exp_name are required." >&2
    echo "Usage: bash scripts/eval/eval_action_flow_solvers.sh <policy> <task> <exp_name> [options]" >&2
    exit 1
fi

POLICY="$1"
TASK="$2"
EXP_NAME="$3"
shift 3

case "$POLICY" in
    action_flow|ddp/action_flow) ;;
    *)
        echo "Error: policy must be an ActionFlow config, got: $POLICY" >&2
        exit 1
        ;;
esac

EPISODES=25
CKPT_TAG="best"
EMA_ARG="--ema"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --ckpt-tag)
            CKPT_TAG="$2"
            shift 2
            ;;
        --no-ema)
            EMA_ARG="--no-ema"
            shift
            ;;
        *)
            echo "Error: unexpected argument: $1" >&2
            exit 1
            ;;
    esac
done

EXP_DIR="experiments/${POLICY}/${TASK}/${EXP_NAME}"
if [[ ! -f "$EXP_DIR/config.yaml" ]]; then
    echo "Error: experiment config not found: $EXP_DIR/config.yaml" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate policy

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$EXP_DIR/eval_dexsim/solver_ablation/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

SOLVERS=(euler midpoint euler midpoint midpoint)
NFES=(2 2 4 4 8)

for index in "${!SOLVERS[@]}"; do
    solver="${SOLVERS[$index]}"
    nfe="${NFES[$index]}"
    label="${solver}_nfe${nfe}"

    echo ""
    echo "============================================================"
    echo "  ActionFlow paired eval: $label"
    echo "  checkpoint=$CKPT_TAG episodes=$EPISODES $EMA_ARG"
    echo "============================================================"

    python dexmani_policy/eval_best_ckpt.py \
        --policy-name="$POLICY" \
        --task-name="$TASK" \
        --exp-name="$EXP_NAME" \
        --ckpt-tag="$CKPT_TAG" \
        --episodes="$EPISODES" \
        --denoise-steps="$nfe" \
        "$EMA_ARG" \
        --no-videos \
        "agent.solver=$solver"

    cp "$EXP_DIR/eval_dexsim/_result.txt" "$OUTPUT_DIR/${label}_result.txt"
    cp "$EXP_DIR/eval_dexsim/result_details.json" "$OUTPUT_DIR/${label}_details.json"
done

echo ""
echo "Paired solver ablation complete: $OUTPUT_DIR"
