#!/usr/bin/env bash
if [[ $# -eq 0 || $1 == -h || $1 == --help ]]; then
    printf 'Usage: %s <config_name> [hydra_overrides...]\n' "$0"
    [[ $# -gt 0 ]] && exit 0
    exit 1
fi

config_name=$1
shift
CDPATH= cd -- "$(dirname -- "$0")/../.." || exit
exec conda run --no-capture-output -n policy \
    python -u -m dexmani_policy.train --config-name="$config_name" "$@"
