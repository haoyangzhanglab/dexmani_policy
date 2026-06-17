#!/usr/bin/env bash
# 下载 Uni3D 预训练权重到 data/pretrained/uni3d/
# 来源: https://huggingface.co/eddie-cui/r3d-weights
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PROJECT_DIR/data/pretrained/uni3d"

if [ -f "$TARGET_DIR/model.safetensors" ]; then
    echo "[download_pretrained] Uni3D weights already exist: $TARGET_DIR/model.safetensors"
    exit 0
fi

echo "[download_pretrained] Downloading Uni3D weights from eddie-cui/r3d-weights..."

mkdir -p "$TARGET_DIR"

# 尝试 huggingface-cli；fallback 到 Python API
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download eddie-cui/r3d-weights model.safetensors \
        --local-dir "$TARGET_DIR"
else
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('eddie-cui/r3d-weights', 'model.safetensors',
                local_dir='$TARGET_DIR')
print('Downloaded to $TARGET_DIR/model.safetensors')
"
fi

echo "[download_pretrained] Done: $TARGET_DIR/model.safetensors"
