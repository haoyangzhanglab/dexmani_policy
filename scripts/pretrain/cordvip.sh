#!/bin/bash
# CORDVIP 观测预训练：cordvip.sh [--output-dir DIR] [--resume-from FILE] [key=value ...]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

exec conda run --no-capture-output -n policy \
    python -u -m dexmani_policy.training.pretrain.obs.cordvip "$@"
