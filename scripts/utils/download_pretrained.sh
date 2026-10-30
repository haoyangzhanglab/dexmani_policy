#!/usr/bin/env bash
# 准备 Uni3D 本地权重与 SAT/DQ-RISE 使用的 DINOv2-small HF 缓存。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_DIR="$PROJECT_DIR/data/pretrained/uni3d"

conda run --no-capture-output -n policy python -c '
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

target_dir = Path(sys.argv[1])
weights = target_dir / "model.safetensors"
if weights.is_file():
    print(f"[download_pretrained] Uni3D weights already exist: {weights}")
else:
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download("eddie-cui/r3d-weights", "model.safetensors", local_dir=target_dir)
    print(f"[download_pretrained] Uni3D: {weights}")

for filename in ("config.json", "model.safetensors"):
    cached = hf_hub_download("facebook/dinov2-small", filename)
    print(f"[download_pretrained] DINOv2-small: {cached}")
' "$TARGET_DIR"
