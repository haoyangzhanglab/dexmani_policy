#!/usr/bin/env bash
# Select a milestone checkpoint using fixed evaluation seeds.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/eval/select_checkpoint.sh <policy_name> <task_name> <exp_name> [args...]"
    echo
    exec conda run --no-capture-output -n policy python dexmani_policy/select_checkpoint.py --help
fi

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/eval/select_checkpoint.sh <policy_name> <task_name> <exp_name> [args...]" >&2
    exit 1
fi

exec conda run --no-capture-output -n policy python dexmani_policy/select_checkpoint.py \
    --policy-name="$1" \
    --task-name="$2" \
    --exp-name="$3" \
    "${@:4}"
