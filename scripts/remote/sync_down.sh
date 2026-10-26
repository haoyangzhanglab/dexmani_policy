#!/usr/bin/env bash
# Download experiments; conflicting configs abort before transfer.
# Usage: bash scripts/remote/sync_down.sh [SUBPATH] [--dry-run|--list] [--with-wandb]
# Transfer weights and run records before updating best.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERVER="${DEX_SERVER:-dexserver}"
REMOTE_EXP="$SERVER:/data_ssd/ZHY/experiments"
LOCAL_EXP="$PROJECT_ROOT/experiments"

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

if [[ -n "$SUBPATH" ]]; then
    if [[ ! "$SUBPATH" =~ ^[a-zA-Z0-9_.-]+(\+[a-zA-Z0-9_.-]+)*(/[a-zA-Z0-9_.-]+(\+[a-zA-Z0-9_.-]+)*)*$ ]] || [[ "$SUBPATH" =~ (^|/)\.\.?(/|$) ]]; then
        echo "Error: invalid SUBPATH '$SUBPATH' (must be a clean relative path like 'dp3/pour')" >&2
        exit 1
    fi
fi

if $LIST_MODE; then
    echo "=== Experiments on server ==="
    if listing=$(ssh "$SERVER" "find /data_ssd/ZHY/experiments -maxdepth 4 -name config.yaml -printf '%h\n'"); then
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

REMOTE_PATH="$REMOTE_EXP${SUBPATH:+/$SUBPATH}"
LOCAL_PATH="$LOCAL_EXP${SUBPATH:+/$SUBPATH}"

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

WANDB_EXCLUDE=()
if ! $WITH_WANDB; then
    WANDB_EXCLUDE=(--exclude='wandb/')
fi

echo "=== sync_down: pulling experiments ==="
echo "  Remote: $REMOTE_PATH"
echo "  Local:  $LOCAL_PATH"
echo ""

# Refuse any config byte difference before transferring experiment artifacts.
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

# --partial would leave incomplete files that --ignore-existing skips on retry.
echo "--- Pass 1/3: new files (--ignore-existing) ---"

PASS1_OPTS=(
    -av
    --ignore-existing
    --progress
    "${WANDB_EXCLUDE[@]}"
    --include='*/'
    --exclude='.dexmani-publish-*'
    --exclude='metrics.jsonl'
    --exclude='checkpoints/latest.pt'
    --exclude='vqvae_hand_best.pt'
    --exclude='prior.pt'
    --exclude='codec.pt'
    --exclude='codebook.npz'
    --exclude='best_ckpt.json'
    --exclude='eval_ckpt_selector/**/selection_result.json'
    --exclude='**/result_details.json'
    $DRY_RUN
)

rsync "${PASS1_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    echo "[sync_down] Pass 1 incomplete (code $rc); mutable references not updated." >&2
    exit "$rc"
}

# rsync's temporary transfer preserves the previous complete file on failure.
echo ""
echo "--- Pass 2/3: training state and selection results ---"

PASS2_OPTS=(
    -av
    "${WANDB_EXCLUDE[@]}"
    --include='metrics.jsonl'
    --include='checkpoints/latest.pt'
    --include='vqvae_hand_best.pt'
    --include='prior.pt'
    --include='codec.pt'
    --include='codebook.npz'
    --include='*/'
    --exclude='*'
    $DRY_RUN
)

rsync "${PASS2_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    echo "[sync_down] Pass 2: rsync error (code $rc)" >&2
    exit "$rc"
}

# Run records may change at completion even if their size and mtime match.
rsync -av --ignore-times "${WANDB_EXCLUDE[@]}" \
    --include='*/' --include='**/result_details.json' \
    --include='eval_ckpt_selector/**/selection_result.json' \
    --exclude='*' $DRY_RUN "$REMOTE_PATH" "$LOCAL_PATH" || exit "$?"

# Copy the small best pointer only after all referenced records succeeded.
echo "--- Pass 3/3: best pointer (--ignore-times) ---"
PASS3_OPTS=(
    -av --ignore-times
    "${WANDB_EXCLUDE[@]}"
    --include='*/'
    --include='best_ckpt.json'
    --exclude='*'
    $DRY_RUN
)
rsync "${PASS3_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    echo "[sync_down] Pass 3: rsync error (code $rc)" >&2
    exit "$rc"
}

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "=== sync_down: done ==="
    echo ""
    echo "Next steps:"
    echo "  bash scripts/eval/select_checkpoint.sh <policy> <task> <exp_name>"
    echo "  bash scripts/eval/evaluate.sh <policy> <task> <exp_name>"
fi
