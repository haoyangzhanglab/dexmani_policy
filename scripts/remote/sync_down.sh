#!/usr/bin/env bash
# Pull one stopped/completed run.
set -euo pipefail

HOST=""
REMOTE_RUN=""
LOCAL_RUN=""
DRY_RUN=()
WITH_WANDB=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="${2:?--host requires a value}"; shift 2 ;;
        --remote-run) REMOTE_RUN="${2:?--remote-run requires an absolute directory}"; shift 2 ;;
        --local-run) LOCAL_RUN="${2:?--local-run requires a directory}"; shift 2 ;;
        --dry-run) DRY_RUN=(--dry-run); shift ;;
        --with-wandb) WITH_WANDB=true; shift ;;
        *) echo "Usage: sync_down.sh --host HOST --remote-run ABS_DIR --local-run DIR [--dry-run] [--with-wandb]" >&2; exit 1 ;;
    esac
done
[[ -n "$HOST" && "$REMOTE_RUN" == /* && "$REMOTE_RUN" != / && -n "$LOCAL_RUN" ]] || { echo "Explicit --host, absolute --remote-run and --local-run are required" >&2; exit 1; }
REMOTE_PATH="$HOST:${REMOTE_RUN%/}/"
LOCAL_PATH="${LOCAL_RUN%/}/"
if [[ ${#DRY_RUN[@]} -eq 0 ]]; then
    mkdir -p "$(dirname "$LOCAL_PATH")"
fi
RSYNC_OPTS=(-avs "${DRY_RUN[@]}")
if ! $WITH_WANDB; then
    RSYNC_OPTS+=(--exclude='wandb/')
fi
printf 'Remote: %s\nLocal: %s\n' "$REMOTE_PATH" "$LOCAL_PATH"

# Keep rsync's default temporary-file transfer for each updated file.
rsync "${RSYNC_OPTS[@]}" --progress \
    --exclude='.dexmani-publish-*' \
    "$REMOTE_PATH" "$LOCAL_PATH"

if [[ ${#DRY_RUN[@]} -eq 0 ]]; then
    echo ""
    echo "=== sync_down: done ==="
    echo ""
fi
