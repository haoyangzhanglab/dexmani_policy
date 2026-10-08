#!/bin/bash
# The validated destination is quoted locally for the remote Bash command.
# shellcheck disable=SC2029
# Sync source to the server; --delete removes stale source files.
# Usage: bash scripts/remote/sync_code.sh [--dry-run] [--dest ABSOLUTE_DIRECTORY]
# A custom destination must be empty; populated launch copies cannot be reused.
# Assets and generated files are excluded; use sync_data.sh / sync_down.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Config ----
SERVER="${DEX_SERVER:-dexserver}"
REMOTE_CODE="$SERVER:~/ZHY/dexmani_policy/"
# ---- End Config ----

DRY_RUN=()
DEST=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN=(--dry-run); shift ;;
        --dest) DEST="${2:?Error: --dest requires an absolute directory}"; shift 2 ;;
        --help|-h)
            echo "Usage: sync_code.sh [--dry-run] [--dest ABSOLUTE_DIRECTORY]"
            echo "Custom destinations must be empty; default mirrors the main source tree."
            exit 0 ;;
        *) echo "Error: unexpected argument '$1'" >&2; exit 1 ;;
    esac
done

if [[ -n "$DEST" ]]; then
    DEST="${DEST%/}"
    if [[ ! "$DEST" =~ ^/[a-zA-Z0-9_.+-]+(/[a-zA-Z0-9_.+-]+)*$ ]] || [[ "$DEST" =~ (^|/)\.\.?(/|$) ]]; then
        echo "Error: --dest must be a clean absolute directory path" >&2
        exit 1
    fi
    printf -v dest_q '%q' "$DEST"
    ssh "$SERVER" "bash -s -- $dest_q" <<'BASH'
set -euo pipefail
dest="$1"
if [[ -e "$dest" && ! -d "$dest" ]]; then
    echo "Error: source destination is not a directory: $dest" >&2
    exit 1
fi
if [[ -d "$dest" ]]; then
    contents="$(find "$dest" -mindepth 1 -maxdepth 1 -print -quit)"
    [[ -z "$contents" ]] || { echo "Error: source destination is already populated: $dest" >&2; exit 1; }
fi
BASH
    REMOTE_CODE="$SERVER:$DEST/"
fi

RSYNC_OPTS=(
    -avz
    --partial
    --progress
    # Recursive (match at any depth — generated files inside packages)
    --exclude='.git/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='*.egg-info'
    --exclude='.DS_Store'
    # Root-anchored (match only at project root — data/symlink dirs)
    --exclude='/.claude'
    --exclude='/.codex'
    --exclude='/.vscode'
    --exclude='/.ruff_cache'
    --exclude='/.mypy_cache'
    --exclude='/.pytest_cache'
    --exclude='/data'
    --exclude='/robot_data'
    --exclude='/experiments'
    --exclude='/pretrained_models'
    --exclude='/wandb'
    --exclude='/_wandb'
    --exclude='/outputs'
    --exclude='/logs'
    --exclude='/bin'
    --exclude='/results'
    # Protect symlink dirs from --delete (they exist on server but not locally)
    --filter='protect /data'
    --filter='protect /robot_data'
    --filter='protect /experiments'
    --delete
    "${DRY_RUN[@]}"
)

echo "=== sync_code: $(basename "$PROJECT_ROOT") → $REMOTE_CODE ==="
rsync "${RSYNC_OPTS[@]}" "$PROJECT_ROOT/" "$REMOTE_CODE"

if [[ ${#DRY_RUN[@]} -eq 0 ]]; then
    echo "=== sync_code: done ==="
fi
