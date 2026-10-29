#!/usr/bin/env bash
# Select a milestone checkpoint using fixed evaluation seeds.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/eval/select_checkpoint.sh <experiment> [args...]"
    echo
    exec conda run --no-capture-output -n policy python -m dexmani_policy.select_checkpoint --help
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/eval/select_checkpoint.sh <experiment> [args...]" >&2
    exit 1
fi

exec conda run --no-capture-output -n policy python -m dexmani_policy.select_checkpoint \
    --experiment="$1" \
    "${@:2}"
