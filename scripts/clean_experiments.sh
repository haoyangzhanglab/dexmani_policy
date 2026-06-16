#!/usr/bin/env bash
# Clean up incomplete and toy experiments under experiments/.
#
# Three categories of interest:
#   A. Incomplete — checkpoint exists but epoch < num_epochs-1, or no checkpoint at all
#   B. Toy        — num_epochs < TOY_MIN_EPOCHS (test/smoke runs), regardless of completion
#
# Usage:
#   bash scripts/clean_experiments.sh                 # dry-run
#   bash scripts/clean_experiments.sh --force         # delete class A; confirm each class B
#   bash scripts/clean_experiments.sh --force --yes   # delete everything without prompts
#   bash scripts/clean_experiments.sh --force --older-than 7
#   bash scripts/clean_experiments.sh --force --include-active
#   bash scripts/clean_experiments.sh --help
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments"

FORCE=false
YES_ALL=false
OLDER_THAN=0
INCLUDE_ACTIVE=false
SKIP_ACTIVE_MINUTES=60
TOY_MIN_EPOCHS=200

show_help() {
    cat <<EOF
Usage: bash scripts/clean_experiments.sh [flags]

Flags:
  --force               Actually delete (default is dry-run).
  --yes, -y             Skip per-item confirmation for class B (toy experiments).
  --older-than DAYS     Only consider experiments older than DAYS.
  --include-active      Include experiments modified within SKIP_ACTIVE_MINUTES.
  --skip-active-min N   Override active threshold (default: ${SKIP_ACTIVE_MINUTES} min).
  --toy-min-epochs N    Override toy threshold (default: ${TOY_MIN_EPOCHS} epochs).
  --help, -h            Show this message.

Examples:
  bash scripts/clean_experiments.sh
  bash scripts/clean_experiments.sh --force
  bash scripts/clean_experiments.sh --force --yes
  bash scripts/clean_experiments.sh --force --older-than 7
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)            FORCE=true ;;
        --yes|-y)           YES_ALL=true ;;
        --older-than)       OLDER_THAN="$2"; shift ;;
        --include-active)   INCLUDE_ACTIVE=true ;;
        --skip-active-min)  SKIP_ACTIVE_MINUTES="$2"; shift ;;
        --toy-min-epochs)   TOY_MIN_EPOCHS="$2"; shift ;;
        --help|-h)          show_help ;;
        *) echo "Unknown arg: $1 (use --help for usage)"; exit 1 ;;
    esac
    shift
done

if [[ ! -d "$EXP_DIR" ]]; then
    echo "experiments/ directory not found"
    exit 0
fi

# ── helpers ──

# Return 0 (true) if the experiment was modified recently enough to be
# considered "active" (still training).
is_active() {
    local exp_dir="$1"
    $INCLUDE_ACTIVE && return 1  # --include-active overrides protection
    local now newest_ts age_minutes
    now=$(date +%s)
    newest_ts=$(find "$exp_dir" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)
    [[ -z "$newest_ts" ]] && return 1
    age_minutes=$(( (now - newest_ts) / 60 ))
    [[ $age_minutes -lt $SKIP_ACTIVE_MINUTES ]]
}

# Read num_epochs from config.yaml; empty string on failure.
get_num_epochs() {
    python -c "
import sys
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(sys.argv[1])
    print(cfg.training.loop.num_epochs)
except Exception:
    pass
" "$1/config.yaml" 2>/dev/null || true
}

# Scan checkpoint filenames for the maximum completed epoch.
# Returns -1 if no checkpoints found.
get_max_epoch() {
    local ckpt_dir="$1" max=-1 epoch pt_file
    for pt_file in "$ckpt_dir"/epoch=*.pt; do
        [[ -f "$pt_file" ]] || continue
        epoch=$(basename "$pt_file" | grep -oP 'epoch=\K\d+')
        [[ -n "$epoch" ]] && { epoch=$((10#$epoch)); [[ $epoch -gt $max ]] && max=$epoch; }
    done
    echo "$max"
}

# ── scan ──

declare -a INCOMPLETE=()
declare -A INCOMPLETE_REASON=()
declare -a TOY=()
declare -A TOY_INFO=()
declare -a SKIPPED=()
TOTAL_INCOMPLETE_SIZE=0
TOTAL_TOY_SIZE=0

while IFS= read -r -d '' checkpoints_dir; do
    exp_dir="$(dirname "$checkpoints_dir")"
    [[ -f "$exp_dir/config.yaml" ]] || continue

    # Age filter
    if [[ "$OLDER_THAN" -gt 0 ]]; then
        dir_age_days=$(( ($(date +%s) - $(stat -c %Y "$exp_dir")) / 86400 ))
        [[ $dir_age_days -lt $OLDER_THAN ]] && continue
    fi

    # Active protection
    if is_active "$exp_dir"; then
        age_min=$(( ($(date +%s) - $(find "$exp_dir" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)) / 60 ))
        SKIPPED+=("$exp_dir|SKIP: modified ${age_min}min ago (use --include-active to override)")
        continue
    fi

    num_epochs=$(get_num_epochs "$exp_dir")
    max_epoch=$(get_max_epoch "$checkpoints_dir")
    size=$(du -sb "$exp_dir" 2>/dev/null | cut -f1)

    # --- classify ---
    if [[ -n "$num_epochs" ]] && [[ $num_epochs -lt $TOY_MIN_EPOCHS ]]; then
        # Class B: toy experiment
        if [[ $max_epoch -ge 0 ]]; then
            info="num_epochs=$num_epochs, completed $((max_epoch + 1))/$num_epochs epochs (last ckpt: epoch=$max_epoch)"
        else
            info="num_epochs=$num_epochs, no checkpoints"
        fi
        TOY+=("$exp_dir")
        TOY_INFO["$exp_dir"]="$info"
        TOTAL_TOY_SIZE=$((TOTAL_TOY_SIZE + size))
    elif [[ $max_epoch -lt 0 ]]; then
        # Class A: no checkpoints
        INCOMPLETE+=("$exp_dir")
        INCOMPLETE_REASON["$exp_dir"]="no checkpoints (training crashed before first checkpoint)"
        TOTAL_INCOMPLETE_SIZE=$((TOTAL_INCOMPLETE_SIZE + size))
    elif [[ -n "$num_epochs" ]] && [[ $max_epoch -lt $((num_epochs - 1)) ]]; then
        # Class A: did not finish
        INCOMPLETE+=("$exp_dir")
        INCOMPLETE_REASON["$exp_dir"]="only $((max_epoch + 1))/$num_epochs epochs completed (last ckpt: epoch=$max_epoch)"
        TOTAL_INCOMPLETE_SIZE=$((TOTAL_INCOMPLETE_SIZE + size))
    fi
done < <(find "$EXP_DIR" -type d -name checkpoints -print0)

# ── report ──

if [[ ${#INCOMPLETE[@]} -eq 0 ]] && [[ ${#TOY[@]} -eq 0 ]]; then
    echo "No incomplete or toy experiments found."
    [[ ${#SKIPPED[@]} -gt 0 ]] && echo "(${#SKIPPED[@]} experiments skipped — still active)"
    exit 0
fi

action_mode="[DRY RUN]"
$FORCE && action_mode="[DELETE]"

# ── class A: incomplete ──

if [[ ${#INCOMPLETE[@]} -gt 0 ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  %s Class A: incomplete experiments (%d)\n" "$action_mode" "${#INCOMPLETE[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for exp_dir in "${INCOMPLETE[@]}"; do
        size_hr=$(du -sh "$exp_dir" 2>/dev/null | cut -f1)
        rel_path="${exp_dir#$ROOT_DIR/}"
        echo "  $rel_path  ($size_hr)"
        echo "    -> ${INCOMPLETE_REASON[$exp_dir]}"
        if $FORCE; then
            rm -rf "$exp_dir"
            # Clean up empty parent dirs
            parent="$(dirname "$exp_dir")"
            while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
                rmdir "$parent"
                parent="$(dirname "$parent")"
            done
        fi
    done
    inc_hr=$(numfmt --to=iec "$TOTAL_INCOMPLETE_SIZE" 2>/dev/null || echo "${TOTAL_INCOMPLETE_SIZE} bytes")
    if $FORCE; then
        echo "  -> Deleted ${#INCOMPLETE[@]} incomplete experiments, freed $inc_hr."
    else
        echo "  -> Would free $inc_hr."
    fi
fi

# ── class B: toy experiments ──

if [[ ${#TOY[@]} -gt 0 ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  %s Class B: toy experiments (num_epochs < %d, %d total)\n" "$action_mode" "$TOY_MIN_EPOCHS" "${#TOY[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for exp_dir in "${TOY[@]}"; do
        size_hr=$(du -sh "$exp_dir" 2>/dev/null | cut -f1)
        rel_path="${exp_dir#$ROOT_DIR/}"
        echo "  $rel_path  ($size_hr)"
        echo "    -> ${TOY_INFO[$exp_dir]}"
    done
    echo ""

    if $FORCE; then
        if $YES_ALL; then
            for exp_dir in "${TOY[@]}"; do
                rm -rf "$exp_dir"
                parent="$(dirname "$exp_dir")"
                while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
                    rmdir "$parent"
                    parent="$(dirname "$parent")"
                done
            done
            toy_hr=$(numfmt --to=iec "$TOTAL_TOY_SIZE" 2>/dev/null || echo "${TOTAL_TOY_SIZE} bytes")
            echo "  -> Deleted ${#TOY[@]} toy experiments, freed $toy_hr."
        else
            echo "  Confirm each item (y=delete  n=skip  a=delete-all-remaining  q=skip-all-remaining):"
            for ((idx=0; idx<${#TOY[@]}; idx++)); do
                exp_dir="${TOY[$idx]}"
                rel_path="${exp_dir#$ROOT_DIR/}"
                read -r -p "    Delete $rel_path? [y/n/a/q] " ans
                case "$ans" in
                    a|A)
                        # Delete current and all remaining items (by array position, not path comparison).
                        for ((i=idx; i<${#TOY[@]}; i++)); do
                            d="${TOY[$i]}"
                            [[ -d "$d" ]] || continue
                            rm -rf "$d"
                            parent="$(dirname "$d")"
                            while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
                                rmdir "$parent"
                                parent="$(dirname "$parent")"
                            done
                        done
                        echo "    -> Deleted current and all remaining toy experiments."
                        break
                        ;;
                    q|Q)
                        echo "    -> Skipped remaining toy experiments."
                        break
                        ;;
                    y|Y)
                        rm -rf "$exp_dir"
                        parent="$(dirname "$exp_dir")"
                        while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
                            rmdir "$parent"
                            parent="$(dirname "$parent")"
                        done
                        echo "    -> Deleted."
                        ;;
                    *)
                        echo "    -> Skipped."
                        ;;
                esac
            done
        fi
    else
        toy_hr=$(numfmt --to=iec "$TOTAL_TOY_SIZE" 2>/dev/null || echo "${TOTAL_TOY_SIZE} bytes")
        echo "  -> Add --force to delete interactively (frees ~$toy_hr); --yes to delete all."
    fi
fi

# ── skipped (active) experiments ──

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo ""
    echo "━━ ${#SKIPPED[@]} experiment(s) skipped (modified within ${SKIP_ACTIVE_MINUTES}min):"
    for s in "${SKIPPED[@]}"; do
        exp_dir="${s%%|*}"; reason="${s#*|}"
        rel_path="${exp_dir#$ROOT_DIR/}"
        echo "  $rel_path"
        echo "    -> $reason"
    done
fi

if ! $FORCE; then
    echo ""
    echo "Dry-run complete. Add --force to actually delete."
fi
