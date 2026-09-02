#!/bin/bash
# ============================================================================
# train_remote.sh — One-click remote training with pre-flight checks
# ============================================================================
# Usage:
#   bash scripts/remote/train_remote.sh <config> <task> [hydra_overrides...]
#   bash scripts/remote/train_remote.sh --fg <config> <task> [...]
#   bash scripts/remote/train_remote.sh --gpus 0,1,2,3 <config> <task> [...]
#   bash scripts/remote/train_remote.sh --sync-data <config> <task> [...]    # incl. data upload
#
# Examples:
#   bash scripts/remote/train_remote.sh dp3 pour
#   bash scripts/remote/train_remote.sh ddp/maniflow pour
#   bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour
#   bash scripts/remote/train_remote.sh --fg dp3 pour 'training.seed=123'
#   bash scripts/remote/train_remote.sh --sync-data dp3 pour  # first run on new server
#   bash scripts/remote/train_remote.sh --dry-run dp3 pour    # preview only
#
# Pre-flight checks (fail-fast):
#   1. Server reachable
#   2. Code synced (sync_code.sh)
#   3. robot_data exists for this task
#   4. GPU memory available
#   5. Disk space on /data_ssd
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Config ----
SERVER="${DEX_SERVER:-dexserver}"
SERVER_PROJ="~/ZHY/dexmani_policy"
SERVER_DATA="/data_ssd/ZHY"
CONDA_PYTHON="~/.conda/envs/dex_policy/bin/python"
# ---- End Config ----

FOREGROUND=false
DRY_RUN=false
GPU_IDS=""
SYNC_DATA=false

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fg)          FOREGROUND=true; shift ;;
        --dry-run|-n)  DRY_RUN=true; shift ;;
        --sync-data)   SYNC_DATA=true; shift ;;
        --gpus)
            GPU_IDS="${2:?Error: --gpus requires a value (e.g., --gpus 0,1,2,3)}"
            shift 2
            ;;
        -*) echo "Error: unknown option '$1'" >&2; exit 1 ;;
        *) break ;;
    esac
done

# Validate --gpus: comma-separated non-negative integers, no duplicates.
if [[ -n "$GPU_IDS" ]]; then
    if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Error: --gpus must be a comma-separated list of integers (e.g. 0,1,2,3), got '$GPU_IDS'" >&2
        exit 1
    fi
    if echo "$GPU_IDS" | tr ',' '\n' | sort -n | uniq -d | grep -q .; then
        echo "Error: --gpus contains duplicate ids: '$GPU_IDS'" >&2
        exit 1
    fi
fi

# Validate positional args
CONFIG="${1:?Error: specify config name (e.g., dp3, ddp/maniflow)}"
TASK="${2:?Error: specify task name (e.g., pour)}"
shift 2

# Validate config name: alphanumeric, /, _, -, . only
if [[ ! "$CONFIG" =~ ^[a-zA-Z0-9_/.-]+$ ]]; then
    echo "Error: invalid config name '$CONFIG' (allowed: a-z, A-Z, 0-9, /, _, -, .)" >&2
    exit 1
fi

# Validate task name: alphanumeric, _, - only
if [[ ! "$TASK" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: invalid task name '$TASK' (allowed: a-z, A-Z, 0-9, _, -)" >&2
    exit 1
fi

# Validate Hydra overrides: reject shell metacharacters
for _override in "$@"; do
    if [[ "$_override" =~ [\;\'\"\`\|\&\<\>\(\)\{\}\#\!\$\\] ]]; then
        echo "Error: override contains unsafe characters: $_override" >&2
        echo "  Allowed: letters, digits, =, ., _, -, /" >&2
        exit 1
    fi
done

HYDRA_OVERRIDES="task_name=$TASK"
for override in "$@"; do
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES $override"
done

# ---- Build remote command ----
if [[ "$CONFIG" == ddp/* ]]; then
    ENTRY="dexmani_policy/train_ddp.py"
else
    ENTRY="dexmani_policy/train.py"
fi
REMOTE_CMD="cd $SERVER_PROJ && $CONDA_PYTHON $ENTRY --config-name=$CONFIG $HYDRA_OVERRIDES"
if [[ -n "$GPU_IDS" ]]; then
    REMOTE_CMD="export CUDA_VISIBLE_DEVICES=$GPU_IDS && $REMOTE_CMD"
fi

SESSION="dex_${CONFIG//\//_}_${TASK}"

# Include seed in session name so same config+task with different seeds
# can run concurrently without killing each other's tmux session.
# Extract from overrides like 'training.seed=42'.
_seed=""
for _override in "$@"; do
    if [[ "$_override" == training.seed=* ]]; then
        _seed="${_override#training.seed=}"
        break
    fi
done
if [[ -n "$_seed" ]]; then
    if [[ ! "$_seed" =~ ^[0-9]+$ ]]; then
        echo "Error: training.seed must be a positive integer, got: $_seed" >&2
        exit 1
    fi
    SESSION="${SESSION}_s${_seed}"
fi

# ---- Dry-run mode ----
if $DRY_RUN; then
    echo "=== DRY RUN ==="
    echo "Config:    $CONFIG"
    echo "Task:      $TASK"
    echo "Session:   $SESSION"
    echo "Overrides: $HYDRA_OVERRIDES"
    echo "GPU:       ${GPU_IDS:-auto}"
    echo "Command:   $REMOTE_CMD"
    echo ""
    bash "$SCRIPT_DIR/sync_code.sh" --dry-run
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

# 2. Code sync
echo -n "[2/5] Syncing code ... "
bash "$SCRIPT_DIR/sync_code.sh" || { echo "FAIL"; exit 1; }
echo "OK"

# 2b. Data sync (optional, first-time)
if $SYNC_DATA; then
    echo -n "[2b] Syncing data (first-time) ... "
    bash "$SCRIPT_DIR/sync_data.sh" || { echo "FAIL"; exit 1; }
    echo "OK"
fi

# 3. robot_data exists
echo -n "[3/5] Dataset $TASK.zarr ... "
if ! ssh "$SERVER" "test -d $SERVER_DATA/robot_data/${TASK}.zarr"; then
    echo "MISSING"
    echo "ERROR: ${TASK}.zarr not found on server." >&2
    echo "  Run: bash scripts/remote/sync_data.sh robot_data" >&2
    exit 1
fi
echo "OK"

# 4. GPU check
echo "[4/5] GPU status:"
ssh "$SERVER" "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null || echo "  (could not query GPUs)"

# Validate requested GPU ids are in range (best-effort).
if [[ -n "$GPU_IDS" ]]; then
    _gpu_count=$(ssh "$SERVER" "nvidia-smi --query-gpu=count --format=csv,noheader" 2>/dev/null || true)
    if [[ -n "$_gpu_count" && "$_gpu_count" =~ ^[0-9]+$ ]]; then
        _max_id=$((_gpu_count - 1))
        for _gid in $(echo "$GPU_IDS" | tr ',' ' '); do
            if [[ $_gid -gt $_max_id ]]; then
                echo "Error: GPU id $_gid out of range (server has $_gpu_count GPUs: 0..$_max_id)" >&2
                exit 1
            fi
        done
    fi
fi

# 5. Disk space
echo -n "[5/5] Disk space /data_ssd ... "
ssh "$SERVER" "df -h /data_ssd | tail -1 | awk '{print \$4 \" available of \" \$2}'" 2>/dev/null || echo "  (could not query disk)"

echo ""
echo "=== Pre-flight OK ==="
echo ""

# Escape single quotes in session name for safe tmux transport.
# Input validation (above) guarantees no shell metacharacters, but single-quote
# escaping is defense-in-depth for the tmux single-quoted context.
SESSION_SAFE="${SESSION//\'/\'\\\'\'}"

# ═══════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════
if $FOREGROUND; then
    echo "Launching in foreground (Ctrl+C to stop)..."
    ssh -t "$SERVER" "$REMOTE_CMD"
else
    # Kill existing session with same name, then create new one
    ssh "$SERVER" "tmux kill-session -t '$SESSION_SAFE' 2>/dev/null; true"
    # Launch in a detached tmux session that self-destructs on completion.
    # stdout/stderr are redirected to logs/<session>.log (root-anchored `logs/`
    # is excluded from sync_code --delete, and .gitignore already lists it), so a
    # crash traceback survives after the session closes. No trailing `read`, so
    # once training exits — checkpoints already saved, GPU memory freed — tmux
    # destroys the session automatically.
    ssh "$SERVER" "tmux new-session -d -s '$SESSION_SAFE' 'mkdir -p $SERVER_PROJ/logs && { $REMOTE_CMD; _rc=\$?; if [ \"\$_rc\" -eq 0 ]; then echo \"[train_remote] $SESSION finished successfully (exit 0).\"; else echo \"[train_remote] $SESSION FAILED (exit \$_rc).\"; fi; } > $SERVER_PROJ/logs/${SESSION_SAFE}.log 2>&1'"

    echo "╔══════════════════════════════════════════╗"
    echo "║  Training started (tmux: $SESSION)"
    echo "╠══════════════════════════════════════════╣"
    echo "║  Attach:  ssh $SERVER -t tmux attach -t '$SESSION_SAFE'"
    echo "║  Log:     bash scripts/remote/tail_log.sh $CONFIG $TASK"
    echo "║  Console: ssh $SERVER \"tail -f $SERVER_PROJ/logs/${SESSION_SAFE}.log\""
    echo "║  Stop:    bash scripts/remote/stop_remote.sh $SESSION"
    echo "╚══════════════════════════════════════════╝"
fi
