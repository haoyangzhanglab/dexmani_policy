#!/usr/bin/env bash
CDPATH= cd -- "$(dirname -- "$0")/../.." || exit
exec conda run --no-capture-output -n policy \
    python -m dexmani_policy.evaluate "$@"
