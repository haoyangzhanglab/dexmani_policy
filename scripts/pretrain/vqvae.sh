#!/bin/bash
# 同源 DQ-RISE 码本预训练：vqvae.sh --policy-config POLICY_CONFIG [key=value ...]
# 输出目录通过 workspace.output_dir 覆盖。
# 默认从 last 导出 codebook.npz；显式 val 时使用 val lookup_mse best。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# conda run 避免激活脚本受 nounset 影响，并实时输出日志。
exec conda run --no-capture-output -n policy \
    python -u -m dexmani_policy.training.pretrain.hand_prior.vqvae "$@"
