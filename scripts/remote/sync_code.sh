#!/bin/bash
# ============================================================================
# sync_code.sh — Push source code to training server (fast, frequent)
# ============================================================================
# Usage:
#   bash scripts/remote/sync_code.sh              # Sync code (default)
#   bash scripts/remote/sync_code.sh --dry-run    # Preview what would change
#
# Design:
#   - rsync -avz for text compression (3-4x ratio on .py/.yaml)
#   - --delete removes stale files on server that were deleted locally
#   - Excludes data/robot_data/experiments (handled by sync_data.sh / sync_down.sh)
#   - Excludes .git, __pycache__, wandb, outputs, and other generated dirs
#   - ~2-3 seconds for typical code delta over LAN
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Config ----
SERVER="${DEX_SERVER:-dexserver}"
REMOTE_CODE="$SERVER:~/ZHY/dexmani_policy/"
# ---- End Config ----

DRY_RUN=""
case "${1:-}" in
    --dry-run|-n) DRY_RUN="--dry-run" ;;
    "")           : ;;
    *)            echo "Error: unexpected argument '${1:-}' (only --dry-run is accepted)" >&2; exit 1 ;;
esac

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
    $DRY_RUN
)

echo "=== sync_code: $(basename "$PROJECT_ROOT") → $REMOTE_CODE ==="
rsync "${RSYNC_OPTS[@]}" "$PROJECT_ROOT/" "$REMOTE_CODE"

if [[ -z "$DRY_RUN" ]]; then
    echo "=== sync_code: done ==="
fi
