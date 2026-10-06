#!/usr/bin/env bash
# One-shot evaluation pipeline: select_best_ckpt → eval_best_ckpt.
#
# Runs selection and numerical held-out evaluation:
#   1. Select the best checkpoint with a fixed initial stage and optional tie batch.
#   2. Evaluate the best checkpoint on disjoint held-out seeds, without videos.
# Demo recording is a separate explicit record_demo.sh command.
#
# Usage:
#   bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>
#   A valid manifest is required; SEED_MANIFEST overrides the saved config path.
#   All stages share this invocation's selection record.
#
# Examples:
#   bash scripts/eval/eval_pipeline.sh dp3 pour 2026-08-01_12-34-56
#   bash scripts/eval/eval_pipeline.sh maniflow pour 2026-08-04_22-19_42
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ── Usage ────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>"
    echo ""
    echo "One-shot evaluation pipeline: select best ckpt → numerical held-out eval."
    echo "A valid manifest is required; set SEED_MANIFEST to override its saved path."
    echo ""
    echo "Positional args:"
    echo "  policy_name   Policy config name (e.g. dp3, maniflow, sat)"
    echo "  task_name     Task name (e.g. pour, pick_apple_messy)"
    echo "  exp_name      Experiment timestamp/name under experiments/<policy>/<task>/"
    echo ""
    echo "  Demo recording is separate: scripts/eval/record_demo.sh"
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval/eval_pipeline.sh dp3 pour 2026-08-01_12-34-56"
    echo "  bash scripts/eval/eval_pipeline.sh maniflow pour 2026-08-04_22-19_42"
    exit 0
fi

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>" >&2
    exit 1
fi

POLICY="$1"
TASK="$2"
EXP_NAME="$3"
shift 3

if [[ $# -gt 0 ]]; then
    echo "Error: unexpected argument: $1. Pipeline always runs without videos; use record_demo.sh separately." >&2
    exit 1
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

# ── Concurrency lock ─────────────────────────────────────────────────────────
LOCK_FILE="${EXP_DIR}/.pipeline.lock"
# Keep the lock file in place. flock locks its inode, so it releases safely
# when this process exits without stale-PID cleanup races.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Error: another eval pipeline is running for ${EXP_DIR}" >&2
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1/2: Select Best Checkpoint (fixed two-stage selection, no videos)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Step 1/2: Select Best Checkpoint"
echo "  (fixed initial stage plus optional exact-tie batch, no videos)"
echo "============================================================"
echo ""

HANDOFF_DIR="$(mktemp -d "${EXP_DIR}/pipeline_XXXXXXXX")"
SELECTION_RECORD="${HANDOFF_DIR}/selection.json"
MANIFEST_ARGS=()
if [[ -n "${SEED_MANIFEST:-}" ]]; then
    MANIFEST_ARGS+=("eval.seed_manifest=${SEED_MANIFEST}")
fi
conda run --no-capture-output -n policy python dexmani_policy/select_best_ckpt.py \
    --policy-name="$POLICY" \
    --task-name="$TASK" \
    --exp-name="$EXP_NAME" \
    --result-file="$SELECTION_RECORD" \
    --no-videos "${MANIFEST_ARGS[@]}"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2/2: Evaluate Best Checkpoint (held-out seeds)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Step 2/2: Evaluate Best Checkpoint (held-out seeds)"
echo "============================================================"
echo ""

# shellcheck disable=SC2086
conda run --no-capture-output -n policy python dexmani_policy/eval_best_ckpt.py \
    --policy-name="$POLICY" \
    --task-name="$TASK" \
    --exp-name="$EXP_NAME" \
    --selection-record="$SELECTION_RECORD" \
    --no-videos

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  ✅ Pipeline Complete!"
echo "============================================================"
echo "  Experiment  : ${EXP_DIR}"
echo "  Selection   : ${SELECTION_RECORD}"
echo "  Eval result : ${EXP_DIR}/eval_dexsim/<run-id>/_result.txt (exact path printed in Step 2)"
echo "============================================================"
