#!/bin/bash
# ============================================================================
# sync_data.sh — Upload datasets & pretrained weights to server data drive
# ============================================================================
# Usage:
#   bash scripts/remote/sync_data.sh                  # Upload all data (size+mtime)
#   bash scripts/remote/sync_data.sh -c               # Upload all (checksum compare)
#   bash scripts/remote/sync_data.sh robot_data       # Upload robot_data only
#   bash scripts/remote/sync_data.sh --dry-run        # Preview what would transfer
#
# Design:
#   - Default: rsync -av (compare by size + mtime). Fast, usually sufficient.
#     Regenerated datasets get new mtimes → correctly detected.
#   - --checksum / -c: compare by MD5 checksum instead. Slower but catches
#     content changes even when mtimes are identical (e.g. restored from backup).
#     Use when you've regenerated a dataset and want absolute certainty.
#   - No -z: .zarr and .safetensors are already compressed.
#   - No --delete: safety — never delete server data if local copy is partial.
#     Stale chunks from old dataset versions are harmless (not referenced by
#     .zarr metadata); clean up manually on server if space is tight.
#   - Targets /data_ssd/ZHY/ directly (persistent NFS, survives container rebuild).
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Config ----
SERVER="${DEX_SERVER:-dexserver}"
REMOTE_DATA="$SERVER:/data_ssd/ZHY/"
# ---- End Config ----

DRY_RUN=""
CHECKSUM=""
TARGET="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN="--dry-run"; shift ;;
        --checksum|-c) CHECKSUM="--checksum"; shift ;;
        all|robot_data|data) TARGET="$1"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

RSYNC_OPTS=(
    -av
    --partial
    --progress
    $CHECKSUM
    $DRY_RUN
)

if [[ -n "$CHECKSUM" ]]; then
    echo "=== sync_data: uploading to /data_ssd/ZHY/ (checksum mode — slower but exact) ==="
else
    echo "=== sync_data: uploading to /data_ssd/ZHY/ (size+mtime mode) ==="
fi

upload_dir() {
    local local_name="$1"
    local local_path="$PROJECT_ROOT/$local_name"
    local remote_subdir="$2"

    if [[ ! -d "$local_path" ]]; then
        echo "[skip] $local_name/ — not found locally"
        return 0
    fi

    echo "[sync] $local_name/ → /data_ssd/ZHY/$remote_subdir/"
    rsync "${RSYNC_OPTS[@]}" "$local_path/" "$REMOTE_DATA/$remote_subdir/"
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
