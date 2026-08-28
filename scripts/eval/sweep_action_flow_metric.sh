#!/usr/bin/env bash
# ActionFlow metric-geometry (§6.5) hyperparameter sweep: train → eval → report SR.
#
# Each config is a full 100k-step training run (single seed) followed by the
# standard eval pipeline (select best ckpt → eval on all seeds, no videos).
# The success rate is read from the eval output and collected into a summary.
#
# Why: switching ActionFlow to metric xyz (use_metric_xyz=true) leaves the
# geometry hyperparameters (patch_radii, RoPE wavelengths) calibrated in the
# old normalized [-1,1] units. Their metric equivalents are:
#   patch_radii [0.04, 0.08]  -> [0.015, 0.030] m   (mean_scale 2.657)
#   RoPE wave   [0.02, 2.0]   -> [0.0075, 0.753] m
#   coord_noise  0.002        -> already metric (2mm), no conversion
#
# Usage:
#   bash scripts/eval/sweep_action_flow_metric.sh [--phase patch|rope|noise|all]
#                                                 [--task pour] [--seed 42]
#                                                 [--dry-run] [--videos]
#
# Recommended order: patch (most sensitive) → rope → noise (optional).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'HELP'
Usage: bash scripts/eval/sweep_action_flow_metric.sh [options]

Options:
  --phase PHASE    patch | rope | noise | all   (default: patch)
  --task TASK      task name                    (default: pour)
  --seed SEED      training seed                (default: 42)
  --dry-run        print commands without running
  --videos         enable videos in eval (default: --no-videos)

Each entry: train (100k steps) -> eval_pipeline (best ckpt, all seeds) -> log SR.
Summary written to experiments/<policy>/<task>/sweep_<phase>_s<seed>.txt
HELP
}

POLICY="action_flow"
TASK="pour"
SEED=42
PHASE="patch"
DRY_RUN=0
NO_VIDEOS="--no-videos"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --task)  TASK="$2";  shift 2 ;;
        --seed)  SEED="$2";  shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --videos)  NO_VIDEOS=""; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unexpected argument: $1" >&2; usage; exit 1 ;;
    esac
done

case "$PHASE" in
    patch|rope|noise|all) ;;
    *) echo "Error: unknown phase '$PHASE' (expected patch|rope|noise|all)" >&2; exit 1 ;;
esac

# ── Sweep definitions: "<label>|<space-separated hydra overrides>" ────────────
# patch_radii keeps the 2-scale [r_small, r_large=2·r_small] structure.
PATCH_SWEEP=$(cat <<'EOF'
patch_0010_0020|agent.pc_encoder_config.patch_radii=[0.010,0.020]
patch_0015_0030|agent.pc_encoder_config.patch_radii=[0.015,0.030]
patch_0020_0040|agent.pc_encoder_config.patch_radii=[0.020,0.040]
patch_0030_0060|agent.pc_encoder_config.patch_radii=[0.030,0.060]
EOF
)

# RoPE wavelengths: ×0.5 / ×1 / ×2 around the metric equivalent [0.0075, 0.753].
ROPE_SWEEP=$(cat <<'EOF'
rope_0p5x|agent.geo_min_wavelength=0.004 agent.geo_max_wavelength=0.38
rope_1p0x|agent.geo_min_wavelength=0.0075 agent.geo_max_wavelength=0.753
rope_2p0x|agent.geo_min_wavelength=0.015 agent.geo_max_wavelength=1.5
EOF
)

# coord_noise: already metric (2mm); secondary, only if noise is suspected too high.
NOISE_SWEEP=$(cat <<'EOF'
noise_0001|dataset.augmentation_cfg.pc.coord_noise.noise_std=0.001
noise_0002|dataset.augmentation_cfg.pc.coord_noise.noise_std=0.002
noise_0004|dataset.augmentation_cfg.pc.coord_noise.noise_std=0.004
EOF
)

case "$PHASE" in
    patch) SWEEP="$PATCH_SWEEP" ;;
    rope)  SWEEP="$ROPE_SWEEP" ;;
    noise) SWEEP="$NOISE_SWEEP" ;;
    all)   SWEEP="$PATCH_SWEEP"$'\n'"$ROPE_SWEEP"$'\n'"$NOISE_SWEEP" ;;
esac

eval "$(conda shell.bash hook)"
conda activate policy

SUMMARY_FILE="experiments/${POLICY}/${TASK}/sweep_${PHASE}_s${SEED}.txt"
: > "$SUMMARY_FILE"

run_entry() {
    local label="$1" overrides="$2"
    local exp_name="${label}_s${SEED}"
    local exp_dir="experiments/${POLICY}/${TASK}/${exp_name}"

    # Split the overrides string into separate quoted words (handles `[...]` safely).
    local -a ov=()
    read -r -a ov <<< "$overrides"

    echo ""
    echo "============================================================"
    echo "  [${PHASE}] ${label}   (seed=${SEED})"
    echo "  overrides: ${overrides}"
    echo "  exp_dir:   ${exp_dir}"
    echo "============================================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] bash scripts/training/train.sh ${POLICY} ${ov[*]} hydra.run.dir=${exp_dir} training.seed=${SEED}"
        echo "[dry-run] bash scripts/eval/eval_pipeline.sh ${POLICY} ${TASK} ${exp_name} ${NO_VIDEOS}"
        return
    fi

    # 1) Train (100k steps).
    if ! bash scripts/training/train.sh "$POLICY" "${ov[@]}" "hydra.run.dir=${exp_dir}" "training.seed=${SEED}"; then
        echo "  [FAILED] training ${label}" >&2
        echo "${label}  TRAIN_FAILED" >> "$SUMMARY_FILE"
        return
    fi

    # 2) Evaluate best checkpoint.
    # shellcheck disable=SC2086
    if ! bash scripts/eval/eval_pipeline.sh "$POLICY" "$TASK" "$exp_name" $NO_VIDEOS; then
        echo "  [FAILED] eval ${label}" >&2
        echo "${label}  EVAL_FAILED" >> "$SUMMARY_FILE"
        return
    fi

    # 3) Record success rate (single float line in _result.txt).
    local result_file="${exp_dir}/eval_dexsim/_result.txt"
    if [[ -f "$result_file" ]]; then
        local sr
        sr="$(cat "$result_file")"
        echo "  >>> SR = ${sr}"
        echo "${label}  ${sr}" >> "$SUMMARY_FILE"
    else
        echo "  [WARN] eval result not found: ${result_file}" >&2
        echo "${label}  NO_RESULT" >> "$SUMMARY_FILE"
    fi
}

while IFS='|' read -r label overrides; do
    [[ -z "$label" ]] && continue
    run_entry "$label" "$overrides"
done <<< "$SWEEP"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Sweep summary (${PHASE}, seed=${SEED})"
echo "============================================================"
if [[ "$DRY_RUN" -eq 0 ]]; then
    printf "  %-18s %s\n" "label" "success_rate"
    printf "  %-18s %s\n" "-----------------" "------------"
    while read -r label sr; do
        printf "  %-18s %s\n" "$label" "$sr"
    done < "$SUMMARY_FILE"
    echo "------------------------------------------------------------"
    echo "  Summary file: ${SUMMARY_FILE}"
else
    echo "  (dry run — no summary)"
fi
echo "============================================================"
