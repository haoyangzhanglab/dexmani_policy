#!/usr/bin/env bash
# Launch one explicit run from a retained source snapshot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST=""
REPO=""
DATA_ROOT=""
RUN=""
PYTHON=""
GPU=""
FOREGROUND=false
DRY_RUN=false
RESUME=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="${2:?--host requires a value}"; shift 2 ;;
        --repo) REPO="${2:?--repo requires an absolute directory}"; shift 2 ;;
        --data-root) DATA_ROOT="${2:?--data-root requires an absolute directory}"; shift 2 ;;
        --run) RUN="${2:?--run requires an absolute directory}"; shift 2 ;;
        --python) PYTHON="${2:?--python requires an absolute interpreter path}"; shift 2 ;;
        --gpu) GPU="${2:?--gpu requires one GPU id}"; shift 2 ;;
        --fg) FOREGROUND=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
[[ -n "$HOST" && "$REPO" == /* && "$DATA_ROOT" == /* && "$RUN" == /* && "$PYTHON" == /* && "$GPU" =~ ^[0-9]+$ ]] || {
    echo "Usage: train_remote.sh --host HOST --repo ABS_DIR --data-root ABS_DIR --run ABS_DIR --python ABS_FILE --gpu ID [--fg] [--dry-run] [--resume] -- [CONFIG] [ARGS...]" >&2; exit 1;
}
SESSION="dex_$(date +%Y%m%d_%H%M%S)_${RANDOM}_${RANDOM}"
SOURCE="${REPO%/}/outputs/remote_sources/$SESSION"
if $RESUME; then
    COMMAND=(-u -m dexmani_policy.training.common.resume --run "$RUN" "$@" --device cuda:0)
else
    CONFIG="${1:?Specify one fresh method config}"
    shift
    run_value="${RUN//\\/\\\\}"
    run_value="${run_value//\"/\\\"}"
    COMMAND=(-u -m dexmani_policy.train "--config-name=$CONFIG" "$@" "hydra.run.dir=\"$run_value\"" training.device=cuda:0)
fi
printf 'Host: %s\nSource: %s\nRun: %s\nSession: %s\n' "$HOST" "$SOURCE" "$RUN" "$SESSION"
if $DRY_RUN; then
    printf 'Command: '; printf '%q ' "$PYTHON" "${COMMAND[@]}"; printf '\n'
    exit 0
fi
printf -v setup_args '%q ' "$REPO" "$SOURCE"
ssh "$HOST" "bash -s -- $setup_args" <<'BASH'
set -euo pipefail
mkdir -p "$1/outputs/remote_sources"
mkdir "$2"
BASH
bash "$SCRIPT_DIR/sync_code.sh" --host "$HOST" --dest "$SOURCE"
printf -v launch_args '%q ' "$SOURCE" "$DATA_ROOT" "$REPO" "$PYTHON" "$GPU" "$SESSION" "$FOREGROUND" "${COMMAND[@]}"
ssh "$HOST" "bash -s -- $launch_args" <<'BASH'
set -euo pipefail
source_dir="$1"; data_root="$2"; repo="$3"; interpreter="$4"; gpu="$5"; session="$6"; foreground="$7"
shift 7
for name in robot_data data pretrained_models; do
    ln -s -- "$data_root/$name" "$source_dir/$name"
done
cd "$source_dir"
export CUDA_VISIBLE_DEVICES="$gpu"
if [[ "$foreground" == true ]]; then
    exec "$interpreter" "$@"
fi
mkdir -p "$repo/logs"
printf -v command_q '%q ' "$interpreter" "$@"
printf -v log_q '%q' "$repo/logs/$session.log"
printf -v script_q '%q' "exec $command_q > $log_q 2>&1"
tmux -L dexmani_policy new-session -d -s "$session" -c "$source_dir" "bash -c $script_q"
printf 'Log: %s\n' "$repo/logs/$session.log"
BASH
