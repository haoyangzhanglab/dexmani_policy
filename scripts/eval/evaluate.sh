#!/usr/bin/env bash
# Evaluate a saved checkpoint with the Python CLI defaults.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/eval/evaluate.sh <policy_name> <task_name> <exp_name> [args...]"
    echo
    exec conda run --no-capture-output -n policy python dexmani_policy/evaluate.py --help
fi

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/eval/evaluate.sh <policy_name> <task_name> <exp_name> [args...]" >&2
    exit 1
fi

exec conda run --no-capture-output -n policy python dexmani_policy/evaluate.py \
    --policy-name="$1" \
    --task-name="$2" \
    --exp-name="$3" \
    "${@:4}"
