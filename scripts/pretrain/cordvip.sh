#!/usr/bin/env bash
CDPATH= cd -- "$(dirname -- "$0")/../.." || exit
exec conda run --no-capture-output -n policy \
    python -u -m dexmani_policy.training.pretrain.obs.cordvip "$@"
