#!/bin/bash

set -euo pipefail

if ! command -v wandb &> /dev/null; then
    echo "错误: wandb 命令未找到，请先安装 wandb (pip install wandb)" >&2
    exit 1
fi

if ! wandb status &> /dev/null; then
    echo "提示: wandb 未登录，请先运行 'wandb login'" >&2
fi

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/wandb_sync.sh <run_dir|--all> [experiments_root]"
    echo "示例: bash scripts/wandb_sync.sh ./wandb/offline-run-20260401_111839-m6zq0mtq"
    echo "      bash scripts/wandb_sync.sh --all"
    echo "      bash scripts/wandb_sync.sh --all experiments"
    exit 1
fi

if [ "$1" = "--all" ]; then
    ROOT="${2:-experiments}"
    if [ ! -d "$ROOT" ]; then
        echo "错误: 目录未找到: $ROOT" >&2
        exit 1
    fi
    find "$ROOT" -path "*/wandb/offline-run-*" -type d 2>/dev/null | while read -r run_dir; do
        echo "Syncing: $run_dir"
        wandb sync "$run_dir"
    done
else
    wandb sync "$1"
fi
