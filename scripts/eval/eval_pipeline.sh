#!/usr/bin/env bash
# Select a checkpoint, then evaluate disjoint held-out seeds without video.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>"
    echo ""
    echo "One-shot evaluation pipeline: select best ckpt → numerical held-out eval."
    echo ""
    echo "Positional args:"
    echo "  policy_name   Experiment policy path (e.g. dp3, ddp/maniflow); reads saved config.yaml"
    echo "  task_name     Task name (e.g. pour, pick_apple_messy)"
    echo "  exp_name      Experiment timestamp/name under experiments/<policy>/<task>/"
    echo ""
    echo "  Demo recording is separate: scripts/eval/record_demo.sh"
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval/eval_pipeline.sh dp3 pour 2026-08-01_12-34-56"
    echo "  bash scripts/eval/eval_pipeline.sh maniflow pour 2026-08-04_22-19_42"
    exit 0
fi

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>" >&2
    exit 1
fi

POLICY="$1"
TASK="$2"
EXP_NAME="$3"
shift 3

if [[ $# -gt 0 ]]; then
    echo "Error: unexpected argument: $1. Pipeline always runs without videos; use record_demo.sh separately." >&2
    exit 1
fi

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
    --no-videos

echo "Evaluating held-out seeds: ${SELECTION_RECORD}"

conda run --no-capture-output -n policy python dexmani_policy/evaluate.py \
    --policy-name="$POLICY" \
    --task-name="$TASK" \
    --exp-name="$EXP_NAME" \
    --selection-record="$SELECTION_RECORD" \
    --no-videos

echo "Pipeline complete. Selection record: ${SELECTION_RECORD}"
