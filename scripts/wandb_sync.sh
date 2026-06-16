#!/bin/bash
# Sync offline W&B runs to the cloud.
#
# Usage:
#   bash scripts/wandb_sync.sh <run_dir>            # sync a single run
#   bash scripts/wandb_sync.sh --all [root_dir]     # sync all offline runs
#   bash scripts/wandb_sync.sh --dry-run --all      # list runs without syncing
#
# Examples:
#   bash scripts/wandb_sync.sh ./wandb/offline-run-20260401_111839-m6zq0mtq
#   bash scripts/wandb_sync.sh --all
#   bash scripts/wandb_sync.sh --all experiments
#   bash scripts/wandb_sync.sh --dry-run --all
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=false

show_help() {
    cat <<EOF
Usage: bash scripts/wandb_sync.sh [flags] <run_dir|--all>

Flags:
  --all [root]    Sync all offline runs under root/ (default: experiments/).
  --dry-run       List what would be synced without uploading.
  --help, -h      Show this message.

Examples:
  bash scripts/wandb_sync.sh wandb/offline-run-20260401_111839-m6zq0mtq
  bash scripts/wandb_sync.sh --all
  bash scripts/wandb_sync.sh --dry-run --all
EOF
    exit 0
}

# Parse flags
SYNC_TARGET=""
SEARCH_ROOT="experiments"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=true ;;
        --all)
            SYNC_TARGET="--all"
            SEARCH_ROOT="${2:-experiments}"
            [[ "$SEARCH_ROOT" != "--all" ]] || SEARCH_ROOT="experiments"
            ;;
        --help|-h)       show_help ;;
        -*)
            echo "Unknown flag: $1 (use --help for usage)" >&2
            exit 1
            ;;
        *)
            SYNC_TARGET="$1"
            ;;
    esac
    shift
done

if [[ -z "$SYNC_TARGET" ]]; then
    echo "Error: no target specified. Use <run_dir> or --all." >&2
    echo "Usage: bash scripts/wandb_sync.sh --help" >&2
    exit 1
fi

# Check wandb CLI availability
if ! command -v wandb &>/dev/null; then
    echo "Error: 'wandb' CLI not found. Install with: pip install wandb" >&2
    exit 1
fi

# sync_one: sync (or dry-run print) a single run directory
sync_one() {
    local run_dir="$1"
    if $DRY_RUN; then
        local run_id
        run_id=$(basename "$run_dir" | sed 's/^offline-run-//')
        local size
        size=$(du -sh "$run_dir" 2>/dev/null | cut -f1)
        echo "  [dry-run] $run_dir  (id=$run_id, $size)"
    else
        echo "Syncing: $run_dir"
        wandb sync "$run_dir"
    fi
}

if [[ "$SYNC_TARGET" == "--all" ]]; then
    if [[ ! -d "$SEARCH_ROOT" ]]; then
        echo "Error: directory not found: $SEARCH_ROOT" >&2
        exit 1
    fi

    echo "Searching for offline W&B runs under: $SEARCH_ROOT"
    if $DRY_RUN; then
        echo "[dry-run mode — no data will be uploaded]"
    fi
    echo ""

    count=0
    while IFS= read -r -d '' run_dir; do
        sync_one "$run_dir"
        ((count++)) || true
    done < <(find "$SEARCH_ROOT" -path "*/wandb/offline-run-*" -type d -print0 2>/dev/null || true)

    if [[ $count -eq 0 ]]; then
        echo "No offline W&B runs found under $SEARCH_ROOT."
    else
        echo ""
        echo "Found $count offline run(s)."
        if $DRY_RUN; then
            echo "Remove --dry-run to sync."
        fi
    fi
else
    if [[ ! -d "$SYNC_TARGET" ]]; then
        echo "Error: run directory not found: $SYNC_TARGET" >&2
        exit 1
    fi
    sync_one "$SYNC_TARGET"
fi
