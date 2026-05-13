#!/bin/bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "用法: bash scripts/wandb_sync.sh [run_dir|--all]"
    echo "示例: bash scripts/wandb_sync.sh ./wandb/offline-run-20260401_111839-m6zq0mtq"
    exit 1
fi

if [ "$1" = "--all" ]; then
    find experiments -path "*/wandb/offline-run-*" -type d 2>/dev/null | while read run_dir; do
        wandb sync "$run_dir"
    done
else
    wandb sync "$1"
fi
