#!/usr/bin/env bash
# Record viewer videos using the saved demo configuration.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name> [args...]"
    echo
    exec conda run --no-capture-output -n policy python dexmani_policy/record_demo.py --help
fi

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name> [args...]" >&2
    exit 1
fi

for arg in "${@:4}"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        exec conda run --no-capture-output -n policy python dexmani_policy/record_demo.py --help
    fi
done

# SAPIEN's viewer requires X11 or XWayland.
if [[ -z "${DISPLAY:-}" ]]; then
    echo "Error: DISPLAY is not set; demo recording requires X11 or XWayland." >&2
    exit 1
fi

exec conda run --no-capture-output -n policy python dexmani_policy/record_demo.py \
    --policy-name="$1" \
    --task-name="$2" \
    --exp-name="$3" \
    "${@:4}"
