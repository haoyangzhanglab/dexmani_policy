#!/usr/bin/env bash
# Export one experiment through the verified deployment exporter.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/deployment/export.sh <experiment_dir> [options]

Export a Real deployment artifact using the Conda environment 'policy'.

Options (forwarded to dexmani_policy.deployment.export):
  --checkpoint SELECTOR  latest (default), best, 80pct, or a checkpoint filename/path
  --zarr-path PATH       Relocate data; semantics must match the checkpoint snapshot
  --output PATH          New .pt artifact inside the experiment's checkpoints/ directory
  -h, --help             Show this help

Paths:
  Relative experiment and explicit Zarr paths use the caller's working directory.
  Relative checkpoint filenames/paths and --output use experiment/checkpoints/.
  Selected checkpoints must resolve inside experiment/checkpoints/.

Existing artifacts are never overwritten; choose a new --output for another export.
The 'best' selector requires best_ckpt.json and never falls back to 'latest'.
Every export verifies the artifact before updating deployment_latest.pt.
Success prints the exporter's JSON receipt; failures preserve its exit status.
The underlying Python CLI still defaults to 'best'; this script defaults to 'latest'.

Examples:
  bash scripts/deployment/export.sh experiments/policy/task/run
  bash scripts/deployment/export.sh experiments/policy/task/run --checkpoint 80pct
  bash scripts/deployment/export.sh /data/experiment --checkpoint best
  bash scripts/deployment/export.sh /data/experiment --zarr-path /data/task.zarr --output deployment-v2.pt
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

# Keep the caller's working directory so relative input paths retain their meaning.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# argparse uses the last occurrence, so an explicit selector overrides latest.
command_args=(
    conda run --no-capture-output -n policy
    python -m dexmani_policy.deployment.export --checkpoint latest
)
exec "${command_args[@]}" "$@"
