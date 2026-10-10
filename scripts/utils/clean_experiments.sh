#!/usr/bin/env bash
# Read-only experiment inventory. Filenames are evidence, not proof of completion.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) [[ $# -ge 2 ]] || { echo "--root requires a directory" >&2; exit 2; }; EXP_DIR="$2"; shift ;;
        --help|-h)
            echo "Usage: clean_experiments.sh [--root DIRECTORY]"
            echo "Read-only: report configs, checkpoint filenames and directory sizes."
            exit 0 ;;
        *) echo "Unsupported option: $1. This command is read-only; use --help." >&2; exit 2 ;;
    esac
    shift
done
[[ -d "$EXP_DIR" ]] || { echo "No experiment directory: $EXP_DIR"; exit 0; }
while IFS= read -r -d '' config; do
    experiment="${config%/config.yaml}"
    printf '\nExperiment: %s\nConfig: %s\n' "$experiment" "$config"
    du -sh -- "$experiment"
    if [[ -d "$experiment/checkpoints" ]]; then
        find "$experiment/checkpoints" -maxdepth 1 \( -type f -o -type l \) -name '*.pt' -printf '  %f -> %l\n'
    else
        echo "  No checkpoints directory"
    fi
done < <(find "$EXP_DIR" -type f -name config.yaml -print0)
echo "Read-only inventory complete. Checkpoint contents and running state were not verified."
