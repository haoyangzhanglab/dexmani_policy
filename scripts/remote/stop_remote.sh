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
    local rc
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$SERVER" "true"; then
        return 0
    else
        rc=$?
        echo "ERROR: cannot reach '$SERVER'. Training state is unknown (exit $rc)." >&2
        return "$rc"
    fi
}

# Preserve tmux errors before parsing, including command-not-found and SSH failure.
list_sessions() {
    ssh "$SERVER" 'output=$(LC_ALL=C tmux list-sessions 2>&1); rc=$?
if [ "$rc" -eq 0 ]; then printf "%s\n" "$output"; exit 0; fi
case "$output" in
    "no sessions"|"no server running on "*|"error connecting to "*" (No such file or directory)")
        if [ "$rc" -eq 1 ]; then exit 0; fi ;;
esac
printf "%s\n" "$output" >&2
exit "$rc"'
}

# Resolve an exact session using a checked listing; tmux status 1 alone is ambiguous.
has_session() {
    local session="$1" sessions line rc
    if sessions=$(list_sessions); then
        :
    else
        rc=$?
        # Reserve 1 for a successful listing with no exact match.
        [[ $rc -eq 1 ]] && return 2
        return "$rc"
    fi
    while IFS= read -r line; do
        [[ "${line%%:*}" == "$session" ]] && return 0
    done <<< "$sessions"
    return 1
}

# Sends SIGINT once; only explicit --force permits a kill after the wait.
_graceful_stop() {
    local session="$1"
    local rc
    local waited=0

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
    if ssh "$SERVER" "tmux send-keys -t '=$session' C-c"; then
        :
    else
        rc=$?
        echo "ERROR: could not send SIGINT to '$session'." >&2
        return "$rc"
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
            echo "  Session disappeared after ${waited}s; checkpoint completeness is unverified."
            break
        fi
        return "$rc"
    done

    if [[ $waited -ge 30 ]]; then
        if ! $FORCE; then
            echo "ERROR: session still running after 30s; no force-stop requested. Use --force explicitly if needed." >&2
            return 1
        fi
        echo "  Explicit --force: killing the selected tmux session..."
        if ssh "$SERVER" "tmux kill-session -t '=$session'"; then
            :
        else
            rc=$?
            echo "ERROR: could not force-kill '$session'." >&2
            return "$rc"
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

    echo "  Session absent; checkpoint completeness is unverified."
}

FORCE=false
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --force ]]; then FORCE=true; else ARGS+=("$arg"); fi
done
if [[ ${#ARGS[@]} -ne 1 ]]; then
    echo "Usage: stop_remote.sh [--force] <session_name> | --all | --list" >&2
    exit 2
fi
set -- "${ARGS[@]}"
case "${1:-}" in
    --list|-l)
        ensure_server_reachable || exit $?
        echo "=== Active tmux sessions on $SERVER ==="
        if sessions=$(list_sessions); then
            printf "%s\n" "${sessions:-(no active sessions)}"
            :
        else
            rc=$?
            echo "ERROR: could not list sessions on '$SERVER' (exit $rc)." >&2
            exit "$rc"
        fi
        ;;
    --all|-a)
        ensure_server_reachable || exit $?
        echo "Stopping all training sessions on $SERVER..."
        if sessions=$(list_sessions); then
            sessions=$(printf "%s\n" "$sessions" | cut -d: -f1)
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
                [[ "$session" =~ ^dex_[a-zA-Z0-9_.-]+$ ]] || continue
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
            echo "Error: invalid session name '$SESSION'. Use the dex_ session printed by train_remote.sh (letters, digits, _, . and - only)." >&2
            exit 1
        fi
        ensure_server_reachable || exit $?
        echo "Stopping session: $SESSION"
        if _graceful_stop "$SESSION"; then
            echo "Stopped."
        else
            rc=$?
            echo "ERROR: could not confirm '$SESSION' stopped." >&2
            exit "$rc"
        fi
        ;;
esac
