#!/bin/bash
# ============================================================================
# sync_down.sh — Pull experiment results from server (existence-based, robust)
# ============================================================================
# Usage:
#   bash scripts/remote/sync_down.sh                           # All experiments
#   bash scripts/remote/sync_down.sh dp3/pour                  # Specific policy/task
#   bash scripts/remote/sync_down.sh dp3/pour/2026-08-03_12-34 # Specific run
#   bash scripts/remote/sync_down.sh --dry-run                 # Preview what would transfer
#   bash scripts/remote/sync_down.sh --list                    # List experiments on server
#
# Two-pass sync: Pass 1 downloads only new files and protects any existing
# local artifact; Pass 2 updates the three small mutable training entries.
# Pass 1 deliberately does not retain partial transfers: --ignore-existing
# would otherwise mistake an interrupted checkpoint for a complete one.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Config ----
SERVER="${DEX_SERVER:-dexserver}"
REMOTE_EXP="$SERVER:/data_ssd/ZHY/experiments"
LOCAL_EXP="$PROJECT_ROOT/experiments"
# ---- End Config ----

DRY_RUN=""
WITH_WANDB=false
SUBPATH=""
LIST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN="--dry-run"; shift ;;
        --with-wandb) WITH_WANDB=true; shift ;;
        --list|-l)    LIST_MODE=true; shift ;;
        -*)           echo "Error: unknown option '$1'" >&2; exit 1 ;;
        *)
            if [[ -n "$SUBPATH" ]]; then
                echo "Error: unexpected extra argument '$1' (SUBPATH already set to '$SUBPATH')" >&2
                exit 1
            fi
            SUBPATH="$1"; shift ;;
    esac
done

# Validate SUBPATH: clean relative path (no absolute, .., empty, or bad components).
if [[ -n "$SUBPATH" ]]; then
    if [[ ! "$SUBPATH" =~ ^[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)*$ ]] || [[ "$SUBPATH" =~ (^|/)\.\.(/|$) ]]; then
        echo "Error: invalid SUBPATH '$SUBPATH' (must be a clean relative path like 'dp3/pour')" >&2
        exit 1
    fi
fi

# ---- List mode ----
if $LIST_MODE; then
    echo "=== Experiments on server ==="
    # List run dirs by their config.yaml marker — handles both non-DDP
    # (<policy>/<task>/<ts>) and DDP (ddp/<policy>/<task>/<ts>) depths.
    ssh "$SERVER" "find /data_ssd/ZHY/experiments -maxdepth 5 -name config.yaml -printf '%h\n' 2>/dev/null | sort" || {
        echo "No experiments found or server unreachable."
    }
    exit 0
fi

# ---- Build paths ----
REMOTE_PATH="$REMOTE_EXP${SUBPATH:+/$SUBPATH}"
LOCAL_PATH="$LOCAL_EXP${SUBPATH:+/$SUBPATH}"

# Ensure trailing / for rsync directory semantics
[[ "$REMOTE_PATH" != */ ]] && REMOTE_PATH="$REMOTE_PATH/"
[[ "$LOCAL_PATH" != */ ]] && LOCAL_PATH="$LOCAL_PATH/"

LOCAL_PARENT="$(dirname "$LOCAL_PATH")"
if [[ ! -d "$LOCAL_PARENT" ]]; then
    if [[ -n "$DRY_RUN" ]]; then
        echo "[would create] $LOCAL_PARENT"
    else
        mkdir -p "$LOCAL_PARENT"
    fi
fi

# ---- Wandb handling ----
WANDB_EXCLUDE=()
if ! $WITH_WANDB; then
    WANDB_EXCLUDE=(--exclude='wandb/')
fi

echo "=== sync_down: pulling experiments ==="
echo "  Remote: $REMOTE_PATH"
echo "  Local:  $LOCAL_PATH"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Pass 1: Download new files only
# ═══════════════════════════════════════════════════════════════════
# Do not combine --partial with --ignore-existing: a later sync would skip an
# interrupted checkpoint merely because its partial destination file exists.
echo "--- Pass 1/2: new files (--ignore-existing) ---"

PASS1_OPTS=(
    -av
    --ignore-existing
    --progress
    "${WANDB_EXCLUDE[@]}"
    $DRY_RUN
)

rsync "${PASS1_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    if [[ $rc -eq 24 ]]; then
        echo "[sync_down] Pass 1: some files vanished during transfer (harmless)."
    else
        echo "[sync_down] Pass 1: rsync error (code $rc)" >&2
        exit $rc
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Pass 2: Force-update files that change during training
# ═══════════════════════════════════════════════════════════════════
# --existing and the file filter update only mutable training metadata.
# This pass has no --ignore-existing, so retaining a partial transfer is safe.
echo ""
echo "--- Pass 2/2: mutable training files (--existing) ---"

PASS2_OPTS=(
    -av
    --existing
    --partial
    --include='metrics.jsonl'
    --include='checkpoints/latest.pt'
    --include='*/'
    --exclude='*'
    "${WANDB_EXCLUDE[@]}"
    $DRY_RUN
)

rsync "${PASS2_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    if [[ $rc -eq 24 ]]; then
        echo "[sync_down] Pass 2: some files vanished during transfer (harmless)."
    else
        echo "[sync_down] Pass 2: rsync error (code $rc)" >&2
        exit $rc
    fi
}

# ---- Done ----
if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "=== sync_down: done ==="
    echo ""
    echo "Next steps:"
    echo "  bash scripts/eval/select_best_ckpt.sh <policy> <task> <exp_name>"
    echo "  bash scripts/eval/eval_best_ckpt.sh <policy> <task> <exp_name>"
fi
