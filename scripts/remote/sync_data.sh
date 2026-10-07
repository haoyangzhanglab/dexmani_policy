#!/bin/bash
# Sync data/ and robot_data/ with /data_ssd/ZHY/; checkpoints use sync_down.sh.
# Usage: bash scripts/remote/sync_data.sh [options] [all|data|robot_data]
# Default: push using size+mtime, without compression or destination deletions.
# --checksum/-c compares content; --pull/-P reverses the transfer direction.
# --prune/-p deletes destination-only files in either direction, without a trash bin.
# Preview pruning with --dry-run first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Config ----
SERVER="${DEX_SERVER:-dexserver}"
REMOTE_DATA="$SERVER:/data_ssd/ZHY/"
# ---- End Config ----

DRY_RUN=""
CHECKSUM=""
PRUNE=""
PULL=""
TARGET="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN="--dry-run"; shift ;;
        --checksum|-c) CHECKSUM="--checksum"; shift ;;
        --prune|-p) PRUNE="--delete"; shift ;;
        --pull|-P) PULL="true"; shift ;;
        all|robot_data|data) TARGET="$1"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

RSYNC_OPTS=(
    -av
    --partial
    --progress
    $CHECKSUM
    $PRUNE
    $DRY_RUN
)

# ---- Build header message ----
if [[ -n "$PULL" ]]; then
    echo "=== sync_data: downloading from /data_ssd/ZHY/ → local (pull mode) ==="
else
    echo "=== sync_data: uploading to /data_ssd/ZHY/ (push mode) ==="
fi
if [[ -n "$PRUNE" ]]; then
    if [[ -n "$PULL" ]]; then
        echo "  PRUNE MODE: local files missing on server WILL be deleted."
    else
        echo "  PRUNE MODE: server files missing locally WILL be deleted."
    fi
    if [[ -z "$DRY_RUN" ]]; then
        echo "  TIP: run with --dry-run first to preview deletions."
    fi
fi

upload_dir() {
    local local_name="$1"
    local local_path="$PROJECT_ROOT/$local_name"
    local remote_subdir="$2"

    if [[ -n "$PULL" ]]; then
        # ---- Pull: server → local ----
        if [[ ! -d "$local_path" ]]; then
            if [[ -n "$DRY_RUN" ]]; then
                echo "[would create] $local_name/"
            else
                echo "[mkdir] $local_name/ — creating local directory"
                mkdir -p "$local_path"
            fi
        fi
        echo "[sync] /data_ssd/ZHY/$remote_subdir/ → $local_name/"
        rsync "${RSYNC_OPTS[@]}" "$REMOTE_DATA/$remote_subdir/" "$local_path/"
    else
        # ---- Push: local → server ----
        if [[ ! -d "$local_path" ]]; then
            echo "[skip] $local_name/ — not found locally"
            return 0
        fi
        echo "[sync] $local_name/ → /data_ssd/ZHY/$remote_subdir/"
        rsync "${RSYNC_OPTS[@]}" "$local_path/" "$REMOTE_DATA/$remote_subdir/"
    fi
}

case "$TARGET" in
    all)
        upload_dir "robot_data" "robot_data"
        upload_dir "data" "data"
        ;;
    robot_data)
        upload_dir "robot_data" "robot_data"
        ;;
    data)
        upload_dir "data" "data"
        ;;
esac

if [[ -z "$DRY_RUN" ]]; then
    echo "=== sync_data: done ==="
fi
