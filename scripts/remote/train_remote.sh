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
if $RESUME; then
    SOURCE=""
    REMOTE_SOURCE=""
    COMMAND=(-u -m dexmani_policy.training.resume --run "$RUN" "$@" --device cuda:0)
else
    SOURCE="${REPO%/}/outputs/remote_sources/$SESSION"
    REMOTE_SOURCE="$HOST:$SOURCE"
    CONFIG="${1:?Specify one fresh method config}"
    shift
    run_value="${RUN//\\/\\\\}"
    run_value="${run_value//\"/\\\"}"
    COMMAND=(-u -m dexmani_policy.train "--config-name=$CONFIG" "$@" "hydra.run.dir=\"$run_value\"" training.device=cuda:0)
fi
printf 'Host: %s\nSource: %s\nRun: %s\nSession: %s\n' "$HOST" "${SOURCE:-saved in $RUN/config.yaml}" "$RUN" "$SESSION"
if $DRY_RUN; then
    printf 'Command: '; printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "DEXMANI_POLICY_REMOTE_SOURCE=${REMOTE_SOURCE:-<saved remote_source>}" "$PYTHON" "${COMMAND[@]}"; printf '\n'
    exit 0
fi
if ! $RESUME; then
    bash "$SCRIPT_DIR/sync_code.sh" "$HOST" "$SOURCE"
fi
printf -v launch_args '%q ' "$SOURCE" "$DATA_ROOT" "$REPO" "$PYTHON" "$GPU" "$SESSION" "$FOREGROUND" "$REMOTE_SOURCE" "$RUN" "${COMMAND[@]}"
ssh "$HOST" "bash -s -- $launch_args" <<'BASH'
set -euo pipefail
source_dir="$1"; data_root="$2"; repo="$3"; interpreter="$4"; gpu="$5"; session="$6"; foreground="$7"; remote_source="$8"
run="$9"
shift 9
if [[ -z "$source_dir" ]]; then
    remote_source="$("$interpreter" -c 'import sys; from omegaconf import OmegaConf; print(OmegaConf.load(sys.argv[1]).remote_source)' "$run/config.yaml")"
    source_dir="/${remote_source#*:/}"
else
    for name in robot_data data; do
        ln -s -- "$data_root/$name" "$source_dir/$name"
    done
fi
cd "$source_dir"
command=(env "CUDA_VISIBLE_DEVICES=$gpu" "DEXMANI_POLICY_REMOTE_SOURCE=$remote_source" "$interpreter" "$@")
if [[ "$foreground" == true ]]; then
    exec "${command[@]}"
fi
mkdir -p "$repo/logs"
printf -v command_q '%q ' "${command[@]}"
printf -v log_q '%q' "$repo/logs/$session.log"
printf -v script_q '%q' "exec $command_q > $log_q 2>&1"
tmux -L dexmani_policy new-session -d -s "$session" -c "$source_dir" "bash -c $script_q"
printf 'Log: %s\n' "$repo/logs/$session.log"
BASH
