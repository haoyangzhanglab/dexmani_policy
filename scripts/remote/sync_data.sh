#!/usr/bin/env bash
# Copy one explicit data directory; destination-only files are retained.
set -euo pipefail

HOST=""
LOCAL=""
REMOTE=""
PULL=false
OPTIONS=(-avs --progress)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="${2:?--host requires a value}"; shift 2 ;;
        --local) LOCAL="${2:?--local requires a directory}"; shift 2 ;;
        --remote) REMOTE="${2:?--remote requires an absolute directory}"; shift 2 ;;
        --pull) PULL=true; shift ;;
        --checksum) OPTIONS+=(--checksum); shift ;;
        --dry-run) OPTIONS+=(--dry-run); shift ;;
        *) echo "Usage: sync_data.sh --host HOST --local DIR --remote ABSOLUTE_DIR [--pull] [--checksum] [--dry-run]" >&2; exit 1 ;;
    esac
done
[[ -n "$HOST" && -n "$LOCAL" && "$REMOTE" == /* && "$REMOTE" != / ]] || { echo "Explicit --host, --local and absolute --remote are required" >&2; exit 1; }
if $PULL; then
    rsync "${OPTIONS[@]}" -- "$HOST:${REMOTE%/}/" "${LOCAL%/}/"
else
    [[ -d "$LOCAL" ]] || { echo "Data directory does not exist: $LOCAL" >&2; exit 1; }
    rsync "${OPTIONS[@]}" -- "${LOCAL%/}/" "$HOST:${REMOTE%/}/"
fi
