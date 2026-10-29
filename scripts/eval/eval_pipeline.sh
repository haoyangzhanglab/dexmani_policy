#!/usr/bin/env bash
# Select a checkpoint and evaluate its frozen test plan.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

USAGE="Usage: bash scripts/eval/eval_pipeline.sh <experiment> [args...]"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "$USAGE"
    echo
    exec conda run --no-capture-output -n policy python -m dexmani_policy.eval_pipeline --help
fi

if [[ $# -lt 1 ]]; then
    echo "$USAGE" >&2
    exit 1
fi

exec conda run --no-capture-output -n policy python -m dexmani_policy.eval_pipeline \
    --experiment="$1" \
    "${@:2}"
