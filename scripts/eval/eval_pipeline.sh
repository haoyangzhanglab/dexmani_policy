#!/usr/bin/env bash
# Select a checkpoint, then evaluate disjoint held-out seeds without video.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

USAGE="Usage: bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name> [--seed SEED] [--episodes N]"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "$USAGE"
    echo ""
    echo "One-shot evaluation pipeline: select best ckpt → numerical held-out eval."
    echo ""
    echo "Positional args:"
    echo "  policy_name   Experiment policy path (e.g. dp3, ddp/maniflow); reads saved config.yaml"
    echo "  task_name     Task name (e.g. pour, pick_apple_messy)"
    echo "  exp_name      Experiment timestamp/name under experiments/<policy>/<task>/"
    echo "  --seed SEED   Use the same explicit evaluation seed for selection and held-out eval"
    echo "  --episodes N Final eval episodes per task after excluding selection seeds"
    echo ""
    echo "  Demo recording is separate: scripts/eval/record_demo.sh"
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval/eval_pipeline.sh dp3 pour 2026-08-01_12-34-56 --seed 1024 --episodes 70"
    exit 0
fi

if [[ $# -lt 3 ]]; then
    echo "$USAGE" >&2
    exit 1
fi

POLICY="$1"
TASK="$2"
EXP_NAME="$3"
shift 3

SEED_ARGS=()
EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "$USAGE" >&2
        exit 1
    fi
    case "$1" in
        --seed) SEED_ARGS+=(--seed "$2") ;;
        --episodes)
            if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: --episodes requires a positive integer." >&2
                exit 1
            fi
            EVAL_ARGS+=(--episodes "$2")
            ;;
        *) echo "$USAGE" >&2; exit 1 ;;
    esac
    shift 2
done

EXP_DIR="experiments/${POLICY}/${TASK}/${EXP_NAME}"

if [[ ! -f "$EXP_DIR/config.yaml" ]]; then
    echo "Error: config.yaml not found in ${EXP_DIR}" >&2
    exit 1
fi

LOCK_FILE="${EXP_DIR}/.pipeline.lock"
# Keep the inode so concurrent invocations share the same flock.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Error: another eval pipeline is running for ${EXP_DIR}" >&2
    exit 1
fi

echo "Selecting a checkpoint: ${EXP_DIR}"

mkdir -p "${EXP_DIR}/eval_ckpt_selector"
SELECTION_DIR="$(mktemp -d "${EXP_DIR}/eval_ckpt_selector/pipeline_XXXXXXXX")"
SELECTION_RECORD="${SELECTION_DIR}/selection_result.json"
conda run --no-capture-output -n policy python dexmani_policy/select_checkpoint.py \
    --policy-name="$POLICY" \
    --task-name="$TASK" \
    --exp-name="$EXP_NAME" \
    --output-dir="$SELECTION_DIR" \
    "${SEED_ARGS[@]}" \
    --no-videos

echo "Evaluating held-out seeds: ${SELECTION_RECORD}"

conda run --no-capture-output -n policy python dexmani_policy/evaluate.py \
    --policy-name="$POLICY" \
    --task-name="$TASK" \
    --exp-name="$EXP_NAME" \
    --selection-record="$SELECTION_RECORD" \
    "${SEED_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    --no-videos

echo "Pipeline complete. Selection record: ${SELECTION_RECORD}"
