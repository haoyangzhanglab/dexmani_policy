"""Extract spatial and global RGB features."""

import argparse
from typing import cast

import torch

from dexmani_policy.agents.obs_encoder.rgb.build import RGB_ENCODER_DEFAULTS, build_rgb_encoder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", nargs="?", choices=RGB_ENCODER_DEFAULTS, default="resnet")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, processor = build_rgb_encoder(args.encoder)
    encoder.to(device).eval()
    images = torch.randint(256, (2, 2, 240, 320, 3), dtype=torch.uint8)
    rgb = cast(torch.Tensor, processor.process_images(images)).to(
        device=device, dtype=next(encoder.backbone.parameters()).dtype
    )
    with torch.no_grad():
        features = encoder(rgb)
        feature_map = encoder.tokens_to_feature_map(features["patch_tokens"], rgb.shape[-2:])
    for key, value in {**features, "feature_map": feature_map}.items():
        print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
