#!/bin/bash
# DQ-RISE 码本预训练：vqvae.sh [task_name] [key=value ...]
# 结束后自动保存所选 best 的 codebook.npz；独立 exporter 仅用于重导出。
# 可选传入 dataset.split_manifest；两阶段默认使用相同的显式 split 参数。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

VQ_OVERRIDES=()
if [[ $# -gt 0 && ! "$1" =~ ^- && "$1" != *=* ]]; then
    VQ_OVERRIDES+=("task_name=$1")
    shift
elif [[ -n "${TASK_NAME:-}" ]]; then
    VQ_OVERRIDES+=("task_name=$TASK_NAME")
fi
if [[ -n "${ZARR_PATH:-}" ]]; then
    VQ_OVERRIDES+=("zarr_path=$ZARR_PATH")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    VQ_OVERRIDES+=("vq_vae.output_dir=$OUTPUT_DIR")
fi

# conda run 避免激活脚本受 nounset 影响，并实时输出日志。
exec conda run --no-capture-output -n policy \
    python -u -m dexmani_policy.training.pretrain.hand_prior.vqvae \
    --config dqrise \
    "${VQ_OVERRIDES[@]}" \
    "$@"
