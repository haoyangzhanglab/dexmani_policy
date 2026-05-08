#!/bin/bash
# Wandb 离线结果同步
# 用法: bash scripts/wandb_sync.sh [run_dir|--all]

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/wandb_sync.sh [run_dir|--all]"
    echo ""
    echo "示例:"
    echo "  # 同步单个 run"
    echo "  bash scripts/wandb_sync.sh ./wandb/offline-run-20260401_111839-m6zq0mtq"
    echo ""
    echo "  # 同步所有离线 run"
    echo "  bash scripts/wandb_sync.sh --all"
    echo ""
    echo "注意: 首次使用需先运行 'wandb login' 完成认证"
    exit 1
fi

if [ "$1" = "--all" ]; then
    echo "同步所有离线 run..."
    wandb sync ./wandb --include-offline --sync-all
else
    echo "同步: $1"
    wandb sync "$1"
fi
