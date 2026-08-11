#!/bin/bash
# ============================================================================
# tail_log.sh — Stream training metrics from server in real time
# ============================================================================
# Usage:
#   bash scripts/remote/tail_log.sh <policy> <task>              # Latest run for task
#   bash scripts/remote/tail_log.sh <policy> <task> <timestamp>  # Specific run
#
# Also works for downloaded experiments (looks locally if server unreachable).
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Config ----
SERVER="${DEX_SERVER:-dexserver}"
SERVER_EXP="/data_ssd/ZHY/experiments"
LOCAL_EXP="$PROJECT_ROOT/experiments"
# ---- End Config ----

POLICY="${1:?Usage: tail_log.sh <policy> <task> [timestamp]}"
TASK="${2:?Usage: tail_log.sh <policy> <task> [timestamp]}"
TS="${3:-}"

find_latest() {
    local base="$1"
    ssh "$SERVER" "ls -dt $base/$POLICY/$TASK/*/ 2>/dev/null | head -1" 2>/dev/null
}

# Try server first, then local
LOG_DIR=""
if ssh -o ConnectTimeout=3 -o BatchMode=yes "$SERVER" "echo ok" &>/dev/null 2>&1; then
    if [[ -n "$TS" ]]; then
        LOG_DIR="$SERVER_EXP/$POLICY/$TASK/$TS"
    else
        LOG_DIR=$(find_latest "$SERVER_EXP" || echo "")
    fi
else
    # Fallback: look locally
    if [[ -n "$TS" ]]; then
        LOG_DIR="$LOCAL_EXP/$POLICY/$TASK/$TS"
    else
        LOG_DIR=$(ls -dt "$LOCAL_EXP/$POLICY/$TASK/"*/ 2>/dev/null | head -1 || echo "")
    fi
fi

if [[ -z "$LOG_DIR" ]]; then
    echo "ERROR: No experiment found for $POLICY/$TASK${TS:+ ($TS)}" >&2
    exit 1
fi

LOG_FILE="${LOG_DIR%/}/metrics.jsonl"

echo "Tailing: $LOG_FILE"
echo "Press Ctrl+C to stop."
echo ""

# Try remote tail first, then local
if [[ "$LOG_DIR" == "$SERVER_EXP"* ]] || [[ "$LOG_DIR" == /* && "$LOG_DIR" != "$LOCAL_EXP"* ]]; then
    ssh "$SERVER" "tail -f '$LOG_FILE'"
else
    tail -f "$LOG_FILE"
fi
