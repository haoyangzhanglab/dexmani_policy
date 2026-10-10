#!/bin/bash
# Paths are explicitly quoted on the client before being sent to SSH.
# shellcheck disable=SC2029
# Stream remote metrics; fall back to downloaded experiments if unreachable.
# Usage: bash scripts/remote/tail_log.sh <policy> <task> [run_name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERVER="${DEX_SERVER:-dexserver}"
SERVER_EXP="/data_ssd/ZHY/experiments"
LOCAL_EXP="$PROJECT_ROOT/experiments"

POLICY="${1:?Usage: tail_log.sh <policy> <task> [run_name]}"
TASK="${2:?Usage: tail_log.sh <policy> <task> [run_name]}"
RUN_NAME="${3:-}"
[[ $# -le 3 ]] || { echo "Error: unexpected extra argument" >&2; exit 1; }

# These identify saved experiments, including policies whose config was removed.
if [[ ! "$POLICY" =~ ^[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)*$ ]] || [[ "$POLICY" =~ (^|/)\.\.?(/|$) ]]; then
    echo "Error: invalid policy path '$POLICY'" >&2
    exit 1
fi
if [[ ! "$TASK" =~ ^[a-zA-Z0-9_-]+(\+[a-zA-Z0-9_-]+)*$ ]]; then
    echo "Error: invalid task '$TASK'" >&2
    exit 1
fi
if [[ -n "$RUN_NAME" ]] && { [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_.+-]+$ ]] || [[ "$RUN_NAME" == . || "$RUN_NAME" == .. ]]; }; then
    echo "Error: invalid run name '$RUN_NAME'" >&2
    exit 1
fi

# Run exactly the same query locally and remotely. File mtimes survive rsync -a;
# directory mtimes also change when an old run receives evaluation artifacts.
LATEST_QUERY=$(cat <<'BASH'
set -euo pipefail
base="$1"
[[ -d "$base" ]] || { echo "Error: experiment directory not found: $base" >&2; exit 1; }
query_dir="$(mktemp -d)"
trap 'rm -rf -- "$query_dir"' EXIT
find "$base" -mindepth 1 -maxdepth 1 -type d -print0 > "$query_dir/candidates"
: > "$query_dir/records"
while IFS= read -r -d '' run; do
    name="${run##*/}"
    [[ "$name" =~ ^[a-zA-Z0-9_.+-]+$ ]] || continue
    [[ -f "$run/config.yaml" && -f "$run/metrics.jsonl" ]] || continue
    stamp="$run/config.yaml"
    mtime="$(find -L "$stamp" -maxdepth 0 -printf '%T@')"
    printf '%s\t%s\0' "$mtime" "$name" >> "$query_dir/records"
done < "$query_dir/candidates"
[[ -s "$query_dir/records" ]] || { echo "Error: no experiment with config.yaml and metrics.jsonl under $base" >&2; exit 1; }
LC_ALL=C sort -z -t $'\t' -k1,1nr -k2,2r "$query_dir/records" > "$query_dir/sorted"
IFS= read -r -d '' selected < "$query_dir/sorted"
printf '%s\n' "${selected#*$'\t'}"
BASH
)

LOG_SOURCE=local
BASE="$LOCAL_EXP/$POLICY/$TASK"
if ssh -o ConnectTimeout=3 -o BatchMode=yes "$SERVER" "echo ok" &>/dev/null; then
    LOG_SOURCE=remote
    BASE="$SERVER_EXP/$POLICY/$TASK"
fi

if [[ -z "$RUN_NAME" ]]; then
    if [[ "$LOG_SOURCE" == remote ]]; then
        printf -v base_q '%q' "$BASE"
        RUN_NAME=$(ssh "$SERVER" "bash -s -- $base_q" <<< "$LATEST_QUERY")
    else
        RUN_NAME=$(bash -s -- "$BASE" <<< "$LATEST_QUERY")
    fi
fi
LOG_DIR="$BASE/$RUN_NAME"
LOG_FILE="$LOG_DIR/metrics.jsonl"

if [[ "$LOG_SOURCE" == remote ]]; then
    printf -v log_dir_q '%q' "$LOG_DIR"
    ssh "$SERVER" "test -d $log_dir_q"
else
    [[ -d "$LOG_DIR" ]] || { echo "Error: experiment directory not found: $LOG_DIR" >&2; exit 1; }
fi

echo "Tailing ($LOG_SOURCE): $LOG_FILE"
echo "Press Ctrl+C to stop."
if [[ "$LOG_SOURCE" == remote ]]; then
    printf -v tail_cmd '%q ' tail -F -- "$LOG_FILE"
    ssh "$SERVER" "$tail_cmd"
else
    tail -F -- "$LOG_FILE"
fi
