#!/usr/bin/env bash
# Freeze test seeds and selection capacity, then select and evaluate without video.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

USAGE="Usage: bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name> [args...]"

for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        echo "$USAGE"
        echo
        exec conda run --no-capture-output -n policy python dexmani_policy/eval_pipeline.py --help
    fi
done

if [[ $# -lt 3 ]]; then
    echo "$USAGE" >&2
    exit 1
fi

EXP_DIR="experiments/$1/$2/$3"

if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: experiment directory not found: ${EXP_DIR}" >&2
    exit 1
fi

LOCK_FILE="${EXP_DIR}/.pipeline.lock"
# Keep the inode so concurrent invocations share the same flock.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Error: another eval pipeline is running for ${EXP_DIR}" >&2
    exit 1
fi

exec conda run --no-capture-output -n policy python dexmani_policy/eval_pipeline.py \
    --policy-name="$1" \
    --task-name="$2" \
    --exp-name="$3" \
    "${@:4}"
