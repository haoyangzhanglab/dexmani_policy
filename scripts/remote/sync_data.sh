#!/bin/bash
# ============================================================================
# sync_data.sh — Upload datasets & pretrained weights to server data drive
# ============================================================================
# Usage:
#   bash scripts/remote/sync_data.sh                  # Upload all data (local→server)
#   bash scripts/remote/sync_data.sh -c               # Upload (checksum compare)
#   bash scripts/remote/sync_data.sh robot_data       # Upload robot_data only
#   bash scripts/remote/sync_data.sh --dry-run        # Preview what would transfer
#   bash scripts/remote/sync_data.sh --prune          # Upload + delete server-only files
#   bash scripts/remote/sync_data.sh --pull           # Download server-only files (safe)
#   bash scripts/remote/sync_data.sh --pull --prune   # Download + delete local-only files
#   bash scripts/remote/sync_data.sh --pull --dry-run # Preview what --pull would download
#
# Design:
#   - Default: push local→server, rsync -av (size+mtime). Fast, usually sufficient.
#     Regenerated datasets get new mtimes → correctly detected.
#   - --checksum / -c: compare by MD5 checksum instead. Slower but catches
#     content changes even when mtimes are identical (e.g. restored from backup).
#   - No -z: .zarr and .safetensors are already compressed.
#   - No --delete by default: safety — never delete remote data if local copy is
#     partial. Use --prune to opt in (works in both directions).
#   - --pull / -P: reverse direction (server→local). Downloads files that exist on
#     the server but not locally. Safe by default — never deletes local files.
#     Use case: two-stage training where stage 1 produces artifacts on the server
#     (e.g. DQ-RISE VQ-VAE checkpoints, extracted codebooks) that stage 2 needs locally.
#   - --prune / -p: enables rsync --delete. Push: deletes server files missing
#     locally. Pull: deletes local files missing on server.
#     ALWAYS do --dry-run first — there is no trash bin on either side.
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
            echo "[mkdir] $local_name/ — creating local directory"
            mkdir -p "$local_path"
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
