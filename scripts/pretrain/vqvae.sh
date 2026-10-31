#!/bin/bash
# DQ-RISE 码本预训练：vqvae.sh [key=value ...]
# 结束后自动保存所选 best 的 codebook.npz；独立 exporter 仅用于重导出。
# 同源两阶段传入相同 dataset.split_manifest，主策略仍检查训练回合上限。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# conda run 避免激活脚本受 nounset 影响，并实时输出日志。
exec conda run --no-capture-output -n policy \
    python -u -m dexmani_policy.training.pretrain.hand_prior.vqvae \
    --config pretrain/hand_prior/vqvae \
    "$@"
