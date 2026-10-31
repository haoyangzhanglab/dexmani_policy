#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
exec conda run --no-capture-output -n policy python -m dexmani_policy.evaluate "$@"
