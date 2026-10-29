#!/usr/bin/env bash
# Evaluate a saved checkpoint with the Python CLI defaults.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/eval/evaluate.sh <experiment> [args...]"
    echo
    exec conda run --no-capture-output -n policy python -m dexmani_policy.evaluate --help
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/eval/evaluate.sh <experiment> [args...]" >&2
    exit 1
fi

exec conda run --no-capture-output -n policy python -m dexmani_policy.evaluate \
    --experiment="$1" \
    "${@:2}"
