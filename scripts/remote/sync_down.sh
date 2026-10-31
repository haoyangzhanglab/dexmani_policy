#!/usr/bin/env bash
# Pull one stopped/completed run, publishing its best pointer after its targets.
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
WANDB_EXCLUDE=()
if ! $WITH_WANDB; then
    WANDB_EXCLUDE=(--exclude='wandb/')
fi
printf 'Remote: %s\nLocal: %s\n' "$REMOTE_PATH" "$LOCAL_PATH"

# Refuse any config byte difference before transferring experiment artifacts.
if config_changes=$(rsync -rclsni --existing --out-format='%i %n' \
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
    -avs
    --ignore-existing
    --progress
    "${WANDB_EXCLUDE[@]}"
    --include='*/'
    --exclude='.dexmani-publish-*'
    --exclude='metrics.jsonl'
    --exclude='checkpoints/latest.pt'
    --exclude='resume.pt'
    --exclude='encoder.pt'
    --exclude='vqvae_hand_best.pt'
    --exclude='codebook.npz'
    --exclude='best_ckpt.json'
    --exclude='eval_ckpt_selector/**/selection_result.json'
    --exclude='**/result_details.json'
    "${DRY_RUN[@]}"
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
    -avs
    "${WANDB_EXCLUDE[@]}"
    --include='metrics.jsonl'
    --include='checkpoints/latest.pt'
    --include='resume.pt'
    --include='encoder.pt'
    --include='vqvae_hand_best.pt'
    --include='codebook.npz'
    --include='*/'
    --exclude='*'
    "${DRY_RUN[@]}"
)

rsync "${PASS2_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    echo "[sync_down] Pass 2: rsync error (code $rc)" >&2
    exit "$rc"
}

# Run records may change at completion even if their size and mtime match.
rsync -avs --ignore-times "${WANDB_EXCLUDE[@]}" \
    --include='*/' --include='**/result_details.json' \
    --include='eval_ckpt_selector/**/selection_result.json' \
    --exclude='*' "${DRY_RUN[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || exit "$?"

# Copy the small best pointer only after all referenced records succeeded.
echo "--- Pass 3/3: best pointer (--ignore-times) ---"
PASS3_OPTS=(
    -avs --ignore-times
    "${WANDB_EXCLUDE[@]}"
    --include='*/'
    --include='best_ckpt.json'
    --exclude='*'
    "${DRY_RUN[@]}"
)
rsync "${PASS3_OPTS[@]}" "$REMOTE_PATH" "$LOCAL_PATH" || {
    rc=$?
    echo "[sync_down] Pass 3: rsync error (code $rc)" >&2
    exit "$rc"
}

if [[ ${#DRY_RUN[@]} -eq 0 ]]; then
    echo ""
    echo "=== sync_down: done ==="
    echo ""
fi
