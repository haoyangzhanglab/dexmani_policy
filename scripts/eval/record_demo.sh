#!/usr/bin/env bash
# High-resolution demo video recording from a trained checkpoint.
#
# Uses the SAPIEN viewer (render_mode="human") to capture 1920×1080 video
# frames, ideal for presentations and demo reels.
#
# Usage:
#   bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name> [args...]
#
# Examples:
#   bash scripts/eval/record_demo.sh dp3 pour 2026-08-01_12-34-56
#   bash scripts/eval/record_demo.sh sat pour 2026-08-01_12-34-56 --ckpt-tag 100pct --episodes 10
#   bash scripts/eval/record_demo.sh maniflow pour 2026-08-01_12-34-56 --resolution 3840 2160
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 3 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name> [args...]"
    echo ""
    echo "Positional args:"
    echo "  policy_name   policy config name (e.g. dp3, sat, maniflow)"
    echo "  task_name     task name (e.g. pour, pick_apple_messy)"
    echo "  exp_name      experiment timestamp/name under experiments/<policy>/<task>/"
    echo ""
    echo "Options (record_demo.py):"
    echo "  --ckpt-tag TAG       Checkpoint: best (strict record), latest, 20pct..100pct (default: best)"
    echo "  --episodes N         Number of episodes to record (default: from config)"
    echo "  --seeds S1 S2 ...    Specific seed numbers to record (overrides --episodes)"
    echo "  --output-dir DIR     Output directory (default: exp_dir/demo_videos/)"
    echo "  --resolution W H     Viewer resolution WIDTH HEIGHT (default: 1920 1080)"
    echo "  --fps N              Video FPS override (default: auto-detect from env)"
    echo "  --denoise-steps N    Single inference step count (default: from config)"
    echo "                       To sweep multiple denoise steps, set in config:"
    echo "                         eval.denoise_timesteps_list=[5,10,20]"
    echo "  --ema / --no-ema     Use EMA weights (default: from config)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval/record_demo.sh dp3 pour 2026-08-01_12-34-56"
    echo "  bash scripts/eval/record_demo.sh sat pour 2026-08-01_12-34-56 --episodes 10"
    echo "  bash scripts/eval/record_demo.sh maniflow pour 2026-08-01_12-34-56 --resolution 3840 2160"
    echo "  bash scripts/eval/record_demo.sh dp3 pour 2026-08-01_12-34-56 --seeds 5 12 33 78"
    exit 1
fi

POLICY="$1"
TASK="$2"
EXP_NAME="$3"
shift 3

EXP_DIR="experiments/${POLICY}/${TASK}/${EXP_NAME}"

if [[ ! -d "$EXP_DIR" ]]; then
    echo "Error: experiment directory not found: ${EXP_DIR}" >&2
    exit 1
fi

if [[ ! -f "$EXP_DIR/config.yaml" ]]; then
    echo "Error: config.yaml not found in ${EXP_DIR}" >&2
    exit 1
fi

# Check for display (SAPIEN viewer requires a running X11/Wayland server)
if [[ -z "${DISPLAY:-}" ]]; then
    echo "Error: DISPLAY is not set — demo recording requires a graphical display." >&2
    echo "Run on a machine with a monitor, or set DISPLAY=:0 if the X server is running." >&2
    exit 1
fi

exec conda run --no-capture-output -n policy python dexmani_policy/record_demo.py \
    --policy-name="${POLICY}" \
    --task-name="${TASK}" \
    --exp-name="${EXP_NAME}" \
    "$@"
