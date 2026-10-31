#!/usr/bin/env bash
# Upload a source snapshot into an explicitly named empty directory.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST=""
DEST=""
DRY_RUN=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="${2:?--host requires a value}"; shift 2 ;;
        --dest) DEST="${2:?--dest requires an absolute directory}"; shift 2 ;;
        --dry-run) DRY_RUN=(--dry-run); shift ;;
        *) echo "Usage: sync_code.sh --host HOST --dest ABSOLUTE_DIRECTORY [--dry-run]" >&2; exit 1 ;;
    esac
done
[[ -n "$HOST" && "$DEST" == /* && "$DEST" != / ]] || { echo "Explicit --host and absolute --dest are required" >&2; exit 1; }
printf -v dest_q '%q' "$DEST"
ssh "$HOST" "bash -s -- $dest_q" <<'BASH'
set -euo pipefail
if [[ -e "$1" ]]; then
    [[ -d "$1" && -z "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
        echo "Source destination must be empty: $1" >&2; exit 1;
    }
fi
BASH
rsync -azs --progress "${DRY_RUN[@]}" \
    --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
    --exclude='/.agents/' --exclude='/.codex/' --exclude='/.claude/' --exclude='/.vscode/' \
    --exclude='/.aws/' --exclude='/.env*' \
    --exclude='/.ruff_cache/' --exclude='/.mypy_cache/' --exclude='/.DS_Store' \
    --exclude='/data' --exclude='/robot_data' --exclude='/experiments' \
    --exclude='/pretrained_models' --exclude='/wandb' --exclude='/_wandb' \
    --exclude='/outputs' --exclude='/logs' --exclude='/results' --exclude='/bin' \
    -- "$PROJECT_ROOT/" "$HOST:${DEST%/}/"
