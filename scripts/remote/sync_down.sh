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
# Two-pass sync strategy (no name-based assumptions):
#
#   Pass 1 — "only new files"
#     rsync --ignore-existing
#     → Files that DON'T exist locally are downloaded (new checkpoints, new runs).
#     → Files that DO exist locally are SKIPPED — regardless of their name/path.
#       This includes locally-generated eval outputs, demo videos, etc.
#       No hardcoded directory name assumptions needed.
#
#   Pass 2 — "mutable training files"
#     Targeted rsync for files that change DURING training:
#       - metrics.jsonl          (grows with each log step)
#       - checkpoints/latest.pt  (symlink target changes)
#       - checkpoints/scores.json (top-k tracker updates)
#     These are small text files — re-transfer is cheap.
#
#   Why this is robust:
#     - Checkpoint .pt files are immutable (written once, never modified).
#       --ignore-existing handles them correctly: new ones downloaded,
#       existing ones skipped.
#     - Locally-generated files (eval_dexsim, demo_videos, best_ckpt.json,
#       whatever future eval scripts create) are protected by their mere
#       existence — no need to know their names in advance.
#     - If eval output paths change in a future code update, no sync
#       script change needed.
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
        *)            SUBPATH="$1"; shift ;;
    esac
done

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

mkdir -p "$(dirname "$LOCAL_PATH")"

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
# --ignore-existing: skip any file that already exists on the receiver.
# This is the core robustness mechanism — locally-generated eval artifacts
# (no matter their name) exist locally → automatically protected.
echo "--- Pass 1/2: new files (--ignore-existing) ---"

PASS1_OPTS=(
    -av
    --ignore-existing
    --partial
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
# These files are small (text) and change continuously during training.
# --ignore-existing would skip them (they already exist locally), so we
# do a targeted second pass WITHOUT --ignore-existing.
#
# --existing ensures we only UPDATE files already pulled by pass 1,
# never create new ones (pass 1 already handles new experiment dirs).
#
# File filter explained:
#   --include='metrics.jsonl'           → training metrics (grows)
#   --include='checkpoints/latest.pt'   → symlink to latest checkpoint
#   --include='checkpoints/scores.json' → top-k checkpoint tracker
#   --include='*/'                      → recurse into directories
#   --exclude='*'                       → skip everything else
echo ""
echo "--- Pass 2/2: mutable training files (--existing) ---"

PASS2_OPTS=(
    -av
    --existing
    --partial
    --include='metrics.jsonl'
    --include='checkpoints/latest.pt'
    --include='checkpoints/scores.json'
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
