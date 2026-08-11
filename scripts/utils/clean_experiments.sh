#!/usr/bin/env bash
# Clean up incomplete and toy experiments under experiments/.
#
# Three categories of interest:
#   A. Incomplete — checkpoint exists but steps < total_train_steps, or no checkpoint at all
#   B. Toy        — total_train_steps < TOY_MIN_STEPS (test/smoke runs), regardless of completion
#
# Usage:
#   bash scripts/utils/clean_experiments.sh                 # dry-run
#   bash scripts/utils/clean_experiments.sh --force         # delete class A; confirm each class B
#   bash scripts/utils/clean_experiments.sh --force --yes   # delete everything without prompts
#   bash scripts/utils/clean_experiments.sh --force --older-than 7
#   bash scripts/utils/clean_experiments.sh --force --include-active
#   bash scripts/utils/clean_experiments.sh --help
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments"

FORCE=false
YES_ALL=false
OLDER_THAN=0
INCLUDE_ACTIVE=false
SKIP_ACTIVE_MINUTES=60
TOY_MIN_STEPS=16000

show_help() {
    cat <<EOF
Usage: bash scripts/utils/clean_experiments.sh [flags]

Flags:
  --force               Actually delete (default is dry-run).
  --yes, -y             Skip per-item confirmation for class B (toy experiments).
  --older-than DAYS     Only consider experiments older than DAYS.
  --include-active      Include experiments modified within SKIP_ACTIVE_MINUTES.
  --skip-active-min N   Override active threshold (default: ${SKIP_ACTIVE_MINUTES} min).
  --toy-min-steps N     Override toy threshold (default: ${TOY_MIN_STEPS} steps).
  --help, -h            Show this message.

Examples:
  bash scripts/utils/clean_experiments.sh
  bash scripts/utils/clean_experiments.sh --force
  bash scripts/utils/clean_experiments.sh --force --yes
  bash scripts/utils/clean_experiments.sh --force --older-than 7
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
        --toy-min-steps)    TOY_MIN_STEPS="$2"; shift ;;
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

# Read total_train_steps from config.yaml; fall back to num_epochs (old format).
# Empty string on failure.
get_total_train_steps() {
    python -c "
import sys
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(sys.argv[1])
    loop = cfg.training.loop
    # Try new format first, fallback to old
    steps = loop.get('total_train_steps', None)
    if steps is None and 'num_epochs' in loop:
        # Old config: approximate steps from epochs (assume ~80 steps/epoch)
        steps = loop.num_epochs * 80
    if steps is not None:
        print(int(steps))
except Exception:
    pass
" "$1/config.yaml" 2>/dev/null || true
}

# Scan checkpoint filenames for the maximum completed step.
# Looks for both old (epoch=*step=*) and new (epoch=*step=*milestone=*) patterns.
# Returns -1 if no checkpoints found.
get_max_step() {
    local ckpt_dir="$1" max=-1 step pt_file base
    for pt_file in "$ckpt_dir"/epoch=*.pt; do
        [[ -f "$pt_file" ]] || continue
        base=$(basename "$pt_file")
        step=$(echo "$base" | grep -oP 'step=\K\d+')
        [[ -n "$step" ]] && { step=$((10#$step)); [[ $step -gt $max ]] && max=$step; }
    done
    echo "$max"
}

# Recursively remove empty parent directories up to EXP_DIR.
cleanup_parents() {
    local exp_dir="$1" parent
    parent="$(dirname "$exp_dir")"
    while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] \
        && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
        rmdir "$parent"
        parent="$(dirname "$parent")"
    done
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

    total_steps=$(get_total_train_steps "$exp_dir")
    max_step=$(get_max_step "$checkpoints_dir")
    size=$(du -sb "$exp_dir" 2>/dev/null | cut -f1)

    # --- classify ---
    if [[ -n "$total_steps" ]] && [[ $total_steps -lt $TOY_MIN_STEPS ]]; then
        # Class B: toy experiment
        if [[ $max_step -ge 0 ]]; then
            info="total_train_steps=$total_steps, completed step $max_step/$total_steps"
        else
            info="total_train_steps=$total_steps, no checkpoints"
        fi
        TOY+=("$exp_dir")
        TOY_INFO["$exp_dir"]="$info"
        TOTAL_TOY_SIZE=$((TOTAL_TOY_SIZE + size))
    elif [[ $max_step -lt 0 ]]; then
        # Class A: no checkpoints
        INCOMPLETE+=("$exp_dir")
        INCOMPLETE_REASON["$exp_dir"]="no checkpoints (training crashed before first checkpoint)"
        TOTAL_INCOMPLETE_SIZE=$((TOTAL_INCOMPLETE_SIZE + size))
    elif [[ -n "$total_steps" ]] && [[ $max_step -lt $((total_steps - 1)) ]]; then
        # Class A: did not finish
        INCOMPLETE+=("$exp_dir")
        INCOMPLETE_REASON["$exp_dir"]="only step $max_step/$total_steps completed"
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
            cleanup_parents "$exp_dir"
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
    printf "  %s Class B: toy experiments (total_train_steps < %d, %d total)\n" "$action_mode" "$TOY_MIN_STEPS" "${#TOY[@]}"
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
                cleanup_parents "$exp_dir"
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
                            cleanup_parents "$d"
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
                        cleanup_parents "$exp_dir"
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
