#!/usr/bin/env bash
# One-shot evaluation pipeline: select_best_ckpt → eval_best_ckpt → record_demo.
#
# Runs the full three-stage evaluation workflow with sensible defaults:
#   1. Select the best checkpoint via adaptive elimination (no videos).
#   2. Evaluate the best checkpoint on all 100 seeds (with videos by default).
#   3. Record 5 high-resolution demo videos from the best checkpoint.
#
# Usage:
#   bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name> [--no-videos]
#
# Examples:
#   bash scripts/eval/eval_pipeline.sh dp3 pour 2026-08-01_12-34-56
#   bash scripts/eval/eval_pipeline.sh maniflow_8l_abl pour 2026-08-04_22-19_42 --no-videos
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate policy

# ── Usage ────────────────────────────────────────────────────────────────────
if [[ $# -lt 3 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name> [--no-videos]"
    echo ""
    echo "One-shot evaluation pipeline: select best ckpt → eval on 100 seeds → 5 demo videos."
    echo ""
    echo "Positional args:"
    echo "  policy_name   Policy config name (e.g. dp3, maniflow, maniflow_8l_abl)"
    echo "  task_name     Task name (e.g. pour, pick_apple_messy)"
    echo "  exp_name      Experiment timestamp/name under experiments/<policy>/<task>/"
    echo ""
    echo "Options:"
    echo "  --no-videos   Disable video recording in Step 2 (eval_best_ckpt)."
    echo "                Step 1 (select_best) never records videos."
    echo "                Step 3 (record_demo) always records videos."
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval/eval_pipeline.sh dp3 pour 2026-08-01_12-34-56"
    echo "  bash scripts/eval/eval_pipeline.sh maniflow_8l_abl pour 2026-08-04_22-19_42 --no-videos"
    exit 1
fi

POLICY="$1"
TASK="$2"
EXP_NAME="$3"
shift 3

# ── Optional --no-videos ─────────────────────────────────────────────────────
NO_VIDEOS=""
if [[ "${1:-}" == "--no-videos" ]]; then
    NO_VIDEOS="--no-videos"
    shift
fi

# ── Validate experiment directory ────────────────────────────────────────────
EXP_DIR="experiments/${POLICY}/${TASK}/${EXP_NAME}"

if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: experiment directory not found: ${EXP_DIR}" >&2
    echo "Check that policy_name, task_name, and exp_name are correct." >&2
    exit 1
fi

if [[ ! -f "$EXP_DIR/config.yaml" ]]; then
    echo "Error: config.yaml not found in ${EXP_DIR}" >&2
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1/3: Select Best Checkpoint (adaptive elimination, no videos)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Step 1/3: Select Best Checkpoint"
echo "  (adaptive elimination across all milestones, no videos)"
echo "============================================================"
echo ""

python dexmani_policy/select_best_ckpt.py \
    --policy-name="$POLICY" \
    --task-name="$TASK" \
    --exp-name="$EXP_NAME" \
    --no-videos

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2/3: Evaluate Best Checkpoint (100 seeds)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Step 2/3: Evaluate Best Checkpoint (100 seeds)"
echo "============================================================"
echo ""

# shellcheck disable=SC2086
python dexmani_policy/eval_best_ckpt.py \
    --policy-name="$POLICY" \
    --task-name="$TASK" \
    --exp-name="$EXP_NAME" \
    $NO_VIDEOS

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3/3: Record Demo Videos (5 episodes, non-fatal — needs display)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Step 3/3: Record Demo Videos (5 episodes, 1920×1080)"
echo "============================================================"
echo ""

python dexmani_policy/record_demo.py \
    --policy-name="$POLICY" \
    --task-name="$TASK" \
    --exp-name="$EXP_NAME" \
    || echo "⚠  Demo recording skipped (no display, or environment error)"

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  ✅ Pipeline Complete!"
echo "============================================================"
echo "  Experiment  : ${EXP_DIR}"
echo "  Best ckpt   : ${EXP_DIR}/best_ckpt.json"
echo "  Eval result : ${EXP_DIR}/eval_dexsim/_result.txt"
echo "  Demo videos : ${EXP_DIR}/demo_videos/"
echo "============================================================"
