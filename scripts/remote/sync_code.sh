#!/usr/bin/env bash
# Upload a source snapshot into a new directory.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${1:-}"
DEST="${2:-}"
DEST="${DEST%/}"
[[ $# -eq 2 && -n "$HOST" && "$DEST" == /* && "$DEST" != / ]] || {
    echo "Usage: sync_code.sh HOST ABSOLUTE_DIRECTORY" >&2; exit 1;
}
printf -v destination_q '%q' "$DEST"
ssh "$HOST" "bash -s -- $destination_q" <<'BASH'
set -euo pipefail
mkdir -p -- "$(dirname -- "$1")"
mkdir -- "$1"
BASH
rsync -azs --progress \
    --exclude='.git/' --exclude='__pycache__/' --exclude='*.py[cod]' --exclude='*.so' \
    --exclude='build/' --exclude='dist/' --exclude='*.egg-info/' --exclude='*.egg' \
    --exclude='.cache/' --exclude='.ruff_cache/' --exclude='.mypy_cache/' \
    --exclude='.ipynb_checkpoints/' --exclude='.DS_Store' \
    --exclude='/.agents/' --exclude='/.codex/' --exclude='/.claude/' --exclude='/.vscode/' \
    --exclude='/.aws/' --exclude='/.env*' \
    --exclude='/data' --exclude='/robot_data' --exclude='/experiments' \
    --exclude='/wandb' --exclude='/_wandb' \
    --exclude='/outputs' --exclude='/logs' --exclude='/results' --exclude='/bin' \
    -- "$PROJECT_ROOT/" "$HOST:$DEST/"
