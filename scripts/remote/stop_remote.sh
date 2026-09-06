#!/bin/bash
# Stop remote training tmux sessions.
#
# Usage:
#   bash scripts/remote/stop_remote.sh <session_name>
#   bash scripts/remote/stop_remote.sh --all
#   bash scripts/remote/stop_remote.sh --list

set -euo pipefail

trap 'echo ""; echo "Interrupted — training may still be running. Re-run stop_remote.sh to ensure stop."; exit 1' INT

SERVER="${DEX_SERVER:-dexserver}"

ensure_server_reachable() {
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$SERVER" "true"; then
        echo "ERROR: cannot reach '$SERVER'. Training state is unknown." >&2
        return 1
    fi
}

# Return 0 when present, 1 when absent, and another status for an SSH error.
has_session() {
    local session="$1"
    local rc

    if ssh "$SERVER" "tmux has-session -t '$session' 2>/dev/null"; then
        return 0
    else
        rc=$?
    fi

    if [[ $rc -eq 1 ]]; then
        return 1
    fi

    echo "ERROR: could not inspect tmux session '$session' on '$SERVER' (ssh exit $rc)." >&2
    return "$rc"
}

# Sends SIGINT first, waits up to 30 seconds, and then force-stops if needed.
_graceful_stop() {
    local session="$1"
    local rc
    local waited=0
    local procs

    if has_session "$session"; then
        :
    else
        rc=$?
        if [[ $rc -ne 1 ]]; then
            return "$rc"
        fi
        echo "  (session not found, nothing to stop)"
        return 0
    fi

    echo "  Sending Ctrl+C (SIGINT) to tmux pane..."
    if ! ssh "$SERVER" "tmux send-keys -t '$session' C-c"; then
        echo "ERROR: could not send SIGINT to '$session'." >&2
        return 1
    fi

    while [[ $waited -lt 30 ]]; do
        if has_session "$session"; then
            sleep 2
            waited=$((waited + 2))
            continue
        else
            rc=$?
        fi
        if [[ $rc -eq 1 ]]; then
            echo "  Session exited gracefully after ${waited}s."
            break
        fi
        return "$rc"
    done

    if [[ $waited -ge 30 ]]; then
        echo "  SIGINT timeout, force-killing tmux session..."
        if ! ssh "$SERVER" "tmux kill-session -t '$session'"; then
            echo "ERROR: could not force-kill '$session'." >&2
            return 1
        fi
        if has_session "$session"; then
            echo "ERROR: '$session' still exists after force-kill." >&2
            return 1
        else
            rc=$?
        fi
        if [[ $rc -ne 1 ]]; then
            return "$rc"
        fi
    fi

    echo -n "  GPU memory: "
    if ! procs=$(ssh "$SERVER" \
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null"); then
        echo "unavailable"
        echo "WARNING: session exited, but GPU cleanup could not be verified." >&2
        return 0
    fi
    if [[ -z "$procs" ]]; then
        echo "clean"
    else
        echo ""
        echo "  ⚠  GPU memory still allocated:"
        echo "$procs" | sed 's/^/      /'
        echo "  Investigate on server: nvidia-smi"
    fi
}

case "${1:-}" in
    --list|-l)
        ensure_server_reachable || exit 1
        echo "=== Active tmux sessions on $SERVER ==="
        if ssh "$SERVER" "tmux list-sessions 2>/dev/null"; then
            :
        else
            rc=$?
            if [[ $rc -eq 1 ]]; then
                echo "(no active sessions)"
            else
                echo "ERROR: could not list sessions on '$SERVER'." >&2
                exit "$rc"
            fi
        fi
        ;;
    --all|-a)
        ensure_server_reachable || exit 1
        echo "Stopping all training sessions on $SERVER..."
        if sessions=$(ssh "$SERVER" "tmux list-sessions 2>/dev/null | cut -d: -f1"); then
            :
        else
            rc=$?
            echo "ERROR: could not list sessions on '$SERVER' (ssh exit $rc)." >&2
            exit "$rc"
        fi
        if [[ -z "$sessions" ]]; then
            echo "  (no active sessions)"
        else
            stopped=0
            failed=0
            for session in $sessions; do
                [[ "$session" == dex_* ]] || continue
                echo "  [$session]"
                if _graceful_stop "$session"; then
                    stopped=$((stopped + 1))
                else
                    failed=1
                fi
            done
            if [[ $failed -eq 0 ]]; then
                echo "Stopped $stopped training session(s)."
            else
                echo "ERROR: stopped $stopped session(s), but one or more session states are unknown." >&2
                exit 1
            fi
        fi
        ;;
    "")
        echo "Usage: stop_remote.sh <session_name> | --all | --list" >&2
        exit 1
        ;;
    *)
        SESSION="$1"
        if [[ ! "$SESSION" =~ ^dex_[a-zA-Z0-9_.-]+$ ]]; then
            echo "Error: invalid session name '$SESSION'. Expected format: dex_<config>_<task>[_s<seed>]" >&2
            exit 1
        fi
        ensure_server_reachable || exit 1
        echo "Stopping session: $SESSION"
        if _graceful_stop "$SESSION"; then
            echo "Stopped."
        else
            echo "ERROR: could not confirm '$SESSION' stopped." >&2
            exit 1
        fi
        ;;
esac
