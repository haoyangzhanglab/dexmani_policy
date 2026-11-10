#!/usr/bin/env bash
# 按方法将预训练权重准备到 data/pretrained/。
set -euo pipefail

usage() {
    printf 'Usage: %s [all|dexlatent|sat|dqrise|dp|r3d|dinov3|siglip]\n' "${0##*/}"
}
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
METHOD="${1-all}"
case "$METHOD" in
    all|dexlatent|sat|dqrise|dp|r3d|dinov3|siglip) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_DIR="$PROJECT_DIR/data/pretrained"

conda run --no-capture-output -n policy python -c '
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from huggingface_hub import hf_hub_download

target_dir = Path(sys.argv[1])
method = sys.argv[2]


def download_hf(repo_id, directory, filenames):
    directory = target_dir / directory
    directory.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        destination = directory / filename
        if destination.is_file():
            print(f"[download_pretrained] Already exists: {destination}")
            continue
        hf_hub_download(repo_id, filename, local_dir=directory)
        print(f"[download_pretrained] Downloaded: {destination}")


if method in ("all", "r3d"):
    download_hf("eddie-cui/r3d-weights", "uni3d", ("model.safetensors",))

if method in ("all", "dp"):
    download_hf("facebook/dinov2-small", "dinov2-small", ("config.json", "model.safetensors"))

if method in ("all", "dexlatent", "sat", "dqrise", "dinov3"):
    download_hf("timm/vit_small_patch16_dinov3.lvd1689m", "dinov3-small",
                ("config.json", "model.safetensors", "LICENSE.md"))

if method == "siglip":
    download_hf("google/siglip-base-patch16-224", "siglip-base-patch16-224",
                ("config.json", "preprocessor_config.json", "README.md", "model.safetensors"))

if method in ("all", "dexlatent"):
    weights = target_dir / "dexlatent" / "checkpoint_epoch_1000.pt"
    if weights.is_file():
        print(f"[download_pretrained] DexLatent weights already exist: {weights}")
    else:
        weights.parent.mkdir(parents=True, exist_ok=True)
        source = "https://raw.githubusercontent.com/EmptyBlueBox/DexLatent/502750df34b9fa6a1b56a66481586887ec722f9c/Checkpoints/20260311_225425/checkpoint_epoch_1000.pt"
        with TemporaryDirectory(dir=weights.parent) as temporary:
            pending = Path(temporary) / weights.name
            urlretrieve(source, pending)
            pending.replace(weights)
        print(f"[download_pretrained] DexLatent: {weights}")
' "$TARGET_DIR" "$METHOD"
