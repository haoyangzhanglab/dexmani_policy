#!/bin/bash
# Download experiments; conflicting configs abort before transfer.
# Usage: bash scripts/remote/sync_down.sh [SUBPATH] [--dry-run|--list] [--with-wandb]
# Passes: new immutable files; live metrics/latest/VQ best; checksummed best/selection.
# No pass retains partial files.

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
    if [[ ! "$SUBPATH" =~ ^[a-zA-Z0-9_.-]+(\+[a-zA-Z0-9_.-]+)*(/[a-zA-Z0-9_.-]+(\+[a-zA-Z0-9_.-]+)*)*$ ]] || [[ "$SUBPATH" =~ (^|/)\.\.?(/|$) ]]; then
        echo "Error: invalid SUBPATH '$SUBPATH' (must be a clean relative path like 'dp3/pour')" >&2
        exit 1
    fi
fi

# ---- List mode ----
if $LIST_MODE; then
    echo "=== Experiments on server ==="
    # List run dirs by their config.yaml marker — handles both non-DDP
    # (<policy>/<task>/<ts>) and DDP (ddp/<policy>/<task>/<ts>) depths.
    if listing=$(ssh "$SERVER" "find /data_ssd/ZHY/experiments -maxdepth 5 -name config.yaml -printf '%h\n'"); then
        if [[ -n "$listing" ]]; then
            printf '%s\n' "$listing" | sort
        else
            echo "No experiments found."
        fi
    else
        exit "$?"
    fi
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

# Compare existing training configs before publishing anything.
# Byte differences are conservatively treated as conflicts, including formatting
# changes. This does not authenticate same-named large binary artifacts.
if config_changes=$(rsync -rclni --existing --out-format='%i %n' \
    "${WANDB_EXCLUDE[@]}" --include='*/' \
    --include='config.yaml' --exclude='*' \
    "$REMOTE_PATH" "$LOCAL_PATH"); then
    conflicts=$(printf '%s\n' "$config_changes" | sed -n '/^[<>ch][fL]/p')
    if [[ -n "$conflicts" ]]; then
        printf 'Training config conflict; no artifacts updated:\n%s\n' "$conflicts" >&2
        exit 1
    fi
else
    exit "$?"
fi

# ═══════════════════════════════════════════════════════════════════
# Pass 1: Download new files only
# ═══════════════════════════════════════════════════════════════════
# Do not combine --partial with --ignore-existing: a later sync would skip an
# interrupted checkpoint merely because its partial destination file exists.
echo "--- Pass 1/3: new files (--ignore-existing) ---"

PASS1_OPTS=(
    -av
    --ignore-existing
    --progress
    "${WANDB_EXCLUDE[@]}"
    --include='*/'
    --exclude='.dexmani-publish-*.tmp'
    --exclude='.dexmani-publish-*.tmp.npz'
    --exclude='checkpoints/*.pt.tmp'
    --exclude='checkpoints/latest.tmp.pt'
    --exclude='vqvae_hand_*.pt.tmp'
    --exclude='.best_ckpt.json*.tmp'
    --exclude='.best_ckpt_selection.json*.tmp'
    --exclude='.selection_result.json*.tmp'
    --exclude='.result_details.json*.tmp'
    --exclude='metrics.jsonl'
    --exclude='checkpoints/latest.pt'
    --exclude='vqvae_hand_best.pt'
    --exclude='best_ckpt.json'
    --exclude='eval_ckpt_selector/*/best_ckpt_selection.json'
    $DRY_RUN
)

rsync "${PASS1_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    echo "[sync_down] Pass 1 incomplete (code $rc); mutable references not updated." >&2
    exit "$rc"
}

# ═══════════════════════════════════════════════════════════════════
# Pass 2: Update explicitly mutable training files
# ═══════════════════════════════════════════════════════════════════
# Default rsync temporary-file transfer preserves the old complete destination
# on failure. Targets of latest/best are downloaded before these references.
echo ""
echo "--- Pass 2/3: metrics, latest checkpoint and VQ best ---"

PASS2_OPTS=(
    -av
    "${WANDB_EXCLUDE[@]}"
    --include='metrics.jsonl'
    --include='checkpoints/latest.pt'
    --include='vqvae_hand_best.pt'
    --include='*/'
    --exclude='*'
    $DRY_RUN
)

rsync "${PASS2_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    echo "[sync_down] Pass 2: rsync error (code $rc)" >&2
    exit "$rc"
}

# Small mutable evidence needs content comparison even at equal size/mtime.
# Checkpoints and growing metrics are intentionally excluded from this pass.
echo "--- Pass 3/3: best pointer and selection progress (--checksum) ---"
PASS3_OPTS=(
    -av --checksum
    "${WANDB_EXCLUDE[@]}"
    --include='*/'
    --include='best_ckpt.json'
    --include='eval_ckpt_selector/*/best_ckpt_selection.json'
    --exclude='*'
    $DRY_RUN
)
rsync "${PASS3_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    echo "[sync_down] Pass 3: rsync error (code $rc)" >&2
    exit "$rc"
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
