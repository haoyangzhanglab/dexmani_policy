#!/bin/bash
# SSH commands are built locally with explicitly quoted arguments.
# shellcheck disable=SC2029
# Launch from a private source copy after validating fresh/resume datasets.
# Usage: bash scripts/remote/train_remote.sh [--fg] [--dry-run] [--sync-data]
#          [--gpus ID] <config> <task> [hydra_overrides...]
# Resume task must match saved identity.
# Resource queries must succeed; the requested GPU ID must appear in the server index list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Config ----
SERVER="${DEX_SERVER:-dexserver}"
# HOME in these templates belongs to the remote interpreter.
# shellcheck disable=SC2016
SERVER_PROJ='$HOME/ZHY/dexmani_policy'
# shellcheck disable=SC2016
CONDA_PYTHON='$HOME/.conda/envs/dex_policy/bin/python'
TMUX_SOCKET=dexmani_policy
# ---- End Config ----

FOREGROUND=false
DRY_RUN=false
GPU_ID=""
SYNC_DATA=false

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fg)          FOREGROUND=true; shift ;;
        --dry-run|-n)  DRY_RUN=true; shift ;;
        --sync-data)   SYNC_DATA=true; shift ;;
        --gpus)
            GPU_ID="${2:?Error: --gpus requires one GPU ID (e.g., --gpus 0)}"
            shift 2
            ;;
        -*) echo "Error: unknown option '$1'" >&2; exit 1 ;;
        *) break ;;
    esac
done

if [[ -n "$GPU_ID" && ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: --gpus requires one non-negative integer GPU ID, got '$GPU_ID'" >&2
    exit 1
fi

# Validate positional args
CONFIG="${1:?Error: specify config name (e.g., dp3, maniflow)}"
TASK="${2:?Error: specify task name (e.g., pour)}"
shift 2

# Validate config name as a clean relative Hydra config path.
if [[ ! "$CONFIG" =~ ^[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)*$ ]] || [[ "$CONFIG" =~ (^|/)\.\.?(/|$) ]]; then
    echo "Error: invalid config name '$CONFIG' (allowed: a-z, A-Z, 0-9, /, _, -, .)" >&2
    exit 1
fi

if [[ ! "$TASK" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: invalid task name '$TASK' (allowed: letters, digits, _, -)" >&2
    exit 1
fi

# ---- Build remote command ----
# The remote command crosses SSH and, for background jobs, tmux. Quote every
# argument for the final remote Bash instead of concatenating user input into a
# command string. The outer SSH transport uses POSIX single quotes, while the
# remote interpreter is explicitly Bash so printf %q has defined semantics.
REMOTE_ARGS=(-m dexmani_policy.train "--config-name=$CONFIG" "task_name=$TASK" "$@")
build_remote_commands() {
    REMOTE_CMD="cd \"$SOURCE_DIR\" && \"$CONDA_PYTHON\""
    for remote_arg in "${REMOTE_ARGS[@]}"; do
        printf -v remote_arg_q '%q' "$remote_arg"
        REMOTE_CMD+=" $remote_arg_q"
    done
    # Run this source copy even when the environment has another editable install.
    PREFLIGHT_CMD="cd \"$SOURCE_DIR\" && \"$CONDA_PYTHON\" -m dexmani_policy.training.common.config --check"
    for remote_arg in "${REMOTE_ARGS[@]:2}"; do
        printf -v remote_arg_q '%q' "$remote_arg"
        PREFLIGHT_CMD+=" $remote_arg_q"
    done
    if [[ -n "$GPU_ID" ]]; then
        printf -v gpu_id_q '%q' "$GPU_ID"
        REMOTE_CMD="export CUDA_VISIBLE_DEVICES=$gpu_id_q && $REMOTE_CMD"
    fi
}

remote_bash_invocation() {
    local remote_script="$1"
    local escaped_script="${remote_script//\'/\'\\\'\'}"
    printf "bash -lc '%s'" "$escaped_script"
}

# This runs on the client; CONDA_PYTHON names an interpreter on the server.
if command -v python3 >/dev/null 2>&1; then
    local_python=python3
else
    local_python=python
fi
launch_uuid=$("$local_python" -c 'import uuid; print(uuid.uuid4().hex)')
SESSION="dex_${CONFIG//\//_}_${TASK}_$(date +%Y%m%d_%H%M%S)_${launch_uuid}"

# Append an explicit seed for readability; the launch suffix ensures uniqueness.
_seed=""
for _override in "$@"; do
    if [[ "$_override" == training.seed=* ]]; then
        _seed="${_override#training.seed=}"
        break
    fi
done
if [[ -n "$_seed" ]]; then
    if [[ ! "$_seed" =~ ^[0-9]+$ ]]; then
        echo "Error: training.seed must be a nonnegative integer, got: $_seed" >&2
        exit 1
    fi
    SESSION="${SESSION}_s${_seed}"
fi
SOURCE_TEMPLATE="$SERVER_PROJ/outputs/remote_sources/$launch_uuid"
SOURCE_DIR="$SOURCE_TEMPLATE"

# ---- Dry-run mode ----
if $DRY_RUN; then
    # Resolve remote HOME without creating the source directory. rsync's
    # preview and the custom-destination check are read-only too.
    SOURCE_DIR=$(ssh "$SERVER" "$(remote_bash_invocation "printf '%s\\n' \"$SOURCE_TEMPLATE\"")")
    build_remote_commands
    echo "=== DRY RUN ==="
    echo "Config:    $CONFIG"
    echo "Task:      $TASK"
    echo "Session:   $SESSION"
    echo "Socket:    $TMUX_SOCKET"
    echo "Source:    $SOURCE_DIR (new private copy; shared data/experiments)"
    printf "Overrides:"
    printf " %q" "${REMOTE_ARGS[@]:3}"
    printf "\n"
    echo "GPU:       ${GPU_ID:-auto}"
    echo "Command:   $REMOTE_CMD"
    echo "Datasets:  $PREFLIGHT_CMD"
    echo ""
    bash "$SCRIPT_DIR/sync_code.sh" --dry-run --dest "$SOURCE_DIR"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# Pre-flight checks
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  train_remote: $CONFIG / $TASK"
echo "╚══════════════════════════════════════════╝"
echo ""

# 1. Server reachable
echo -n "[1/5] Server reachable ... "
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$SERVER" "echo ok" &>/dev/null; then
    echo "FAIL"
    echo "ERROR: Cannot reach '$SERVER'. Check VPN/network and SSH config." >&2
    exit 1
fi
echo "OK"

# 2. Reserve a new copy. Never rsync over a source directory used by an old run.
echo -n "[2/5] Preparing private source ... "
PREPARE_SOURCE="set -e; mkdir -p \"$SERVER_PROJ/outputs/remote_sources\"; mkdir \"$SOURCE_TEMPLATE\"; cd \"$SOURCE_TEMPLATE\"; pwd -P"
if SOURCE_DIR=$(ssh "$SERVER" "$(remote_bash_invocation "$PREPARE_SOURCE")"); then
    :
else
    rc=$?
    echo "FAIL" >&2
    echo "ERROR: cannot reserve a new source directory; no training launched." >&2
    exit "$rc"
fi
build_remote_commands
bash "$SCRIPT_DIR/sync_code.sh" --dest "$SOURCE_DIR" || { echo "FAIL"; exit 1; }
echo "OK"
echo "Source: $SOURCE_DIR"

# 2b. Data sync (optional, initial upload or later updates)
if $SYNC_DATA; then
    echo -n "[2b] Syncing data ... "
    bash "$SCRIPT_DIR/sync_data.sh" || { echo "FAIL"; exit 1; }
    echo "OK"
fi

# Relative data and resume paths keep using the existing shared storage. Weight
# roots may be absent for policies that do not use them; their links can dangle.
SHARED_LINKS="set -e; mkdir -p \"$SERVER_PROJ/experiments\" \"$SERVER_PROJ/logs\"; for shared in robot_data data experiments pretrained_models logs; do ln -s \"$SERVER_PROJ/\$shared\" \"$SOURCE_DIR/\$shared\"; done"
ssh "$SERVER" "$(remote_bash_invocation "$SHARED_LINKS")"

# 3. Resolve dataset semantics after code/data sync; do not infer from task_name.
echo "[3/5] Resolved dataset directories ..."
if ! ssh "$SERVER" "$(remote_bash_invocation "$PREFLIGHT_CMD")"; then
    echo "ERROR: dataset pre-flight failed; training was not launched." >&2
    echo "  Check config overrides, persistent-data symlinks, or sync_data.sh robot_data." >&2
    exit 1
fi
echo "OK"

# 4. GPU check
echo "[4/5] GPU status:"
gpu_status=$(ssh "$SERVER" "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
printf '%s\n' "$gpu_status"

# Check the actual indices, including hosts whose index list is not contiguous.
if [[ -n "$GPU_ID" ]]; then
    available_gpu_ids=()
    while IFS=, read -r available_gpu_id gpu_status_remainder; do
        available_gpu_ids+=("$available_gpu_id")
    done <<< "$gpu_status"
    case " ${available_gpu_ids[*]} " in
        *" $GPU_ID "*) ;;
        *)
            echo "Error: GPU id $GPU_ID is unavailable (server GPU indices: ${available_gpu_ids[*]})" >&2
            exit 1
            ;;
    esac
fi

# 5. Disk space
echo -n "[5/5] Disk space /data_ssd ... "
ssh "$SERVER" "df -h /data_ssd"

echo ""
echo "Connection, dataset paths and resource queries succeeded; training requirements still depend on the selected configuration."
echo ""

# ═══════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════
if $FOREGROUND; then
    echo "Launching in foreground (Ctrl+C to stop)..."
    ssh -t "$SERVER" "$(remote_bash_invocation "$REMOTE_CMD")"
else
    printf -v session_q '%q' "$SESSION"
    # Launch in a detached tmux session that self-destructs on completion.
    # stdout/stderr are redirected to logs/<session>.log (root-anchored `logs/`
    # is excluded from sync_code --delete, and .gitignore already lists it), so a
    # crash traceback survives after the session closes. No trailing `read`, so
    # tmux destroys the session when training exits, including on failure.
    # Session exit alone does not establish that a checkpoint was saved.
    REMOTE_TMUX_SCRIPT="set -o noclobber; mkdir -p \"$SERVER_PROJ/logs\" && { $REMOTE_CMD; _rc=\$?; if [[ \$_rc -eq 0 ]]; then echo \"[train_remote] $SESSION finished successfully (exit 0).\"; else echo \"[train_remote] $SESSION FAILED (exit \$_rc).\"; fi; exit \$_rc; } > \"$SERVER_PROJ/logs/${SESSION}.log\" 2>&1"
    printf -v tmux_script_q '%q' "$REMOTE_TMUX_SCRIPT"
    ssh "$SERVER" "$(remote_bash_invocation "tmux -L $TMUX_SOCKET new-session -d -s $session_q bash -lc $tmux_script_q")" || {
        rc=$?
        echo "ERROR: failed to start tmux session '$SESSION' on '$SERVER'." >&2
        exit "$rc"
    }

    echo "╔══════════════════════════════════════════╗"
    echo "║  Launch submitted (tmux: $SESSION)"
    echo "╠══════════════════════════════════════════╣"
    echo "║  Attach:  ssh $SERVER -t tmux -L $TMUX_SOCKET attach -t '$SESSION'"
    echo "║  Console: ssh $SERVER \"tail -f $SERVER_PROJ/logs/${SESSION}.log\""
    echo "║  Stop:    bash scripts/remote/stop_remote.sh $SESSION"
    echo "╚══════════════════════════════════════════╝"
fi
