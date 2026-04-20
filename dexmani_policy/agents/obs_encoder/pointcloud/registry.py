import torch.nn as nn
from typing import Dict, Optional

from dexmani_policy.agents.obs_encoder.pointcloud import (
    MultiStagePointNet,
    PointNet,
    PointNextEncoder,
    PointNextPatchTokenizer,
    PointPNTokenizer,
)


GLOBAL_ENCODER_CONFIGS: Dict[str, Dict] = {
    "dp3": {
        "output_channels": 256,
    },
    "idp3": {
        "output_channels": 256,
    },
    "pointnext": {
        "output_channels": 256,
        "stage_depths": (1, 2, 2),
        "stage_strides": (1, 2, 2),
        "stage_channels": (64, 128, 256),
        "radii": (0.04, 0.08, 0.16),
        "num_neighbors": (24, 24, 32),
    },
}


PATCH_TOKENIZER_CONFIGS: Dict[str, Dict] = {
    "pointpn": {
        "input_points": 1024,
        "num_stages": 3,
        "embed_channels": 64,
        "stage_num_neighbors": (24, 24, 16),
        "stage_lga_blocks": (2, 2, 1),
        "stage_channel_expansion": (2, 2, 2),
        "point_cloud_type": "scan",
    },
    "pointnext_tokenizer": {
        "stem_channels": 64,
        "token_channels": 128,
        "num_patches": 96,
        "patch_radii": (0.04, 0.08),
        "patch_neighbors": (16, 32),
    },
}


def _merge_config(default_cfg: Dict, config: Optional[Dict] = None) -> Dict:
    cfg = dict(default_cfg)
    if config:
        cfg.update(config)
    return cfg


def build_pc_global_encoder(
    encoder_type: str,
    pc_dim: int,
    config: Optional[Dict] = None,
) -> nn.Module:
    if encoder_type not in GLOBAL_ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown global encoder type: {encoder_type}. "
            f"Available types: {sorted(GLOBAL_ENCODER_CONFIGS.keys())}"
        )

    cfg = _merge_config(GLOBAL_ENCODER_CONFIGS[encoder_type], config)

    if encoder_type == "dp3":
        return PointNet(
            input_channels=pc_dim,
            output_channels=cfg["output_channels"],
        )

    if encoder_type == "idp3":
        return MultiStagePointNet(
            input_channels=pc_dim,
            output_channels=cfg["output_channels"],
        )

    if encoder_type == "pointnext":
        return PointNextEncoder(
            input_channels=pc_dim,
            output_channels=cfg["output_channels"],
            stage_depths=cfg["stage_depths"],
            stage_strides=cfg["stage_strides"],
            stage_channels=cfg["stage_channels"],
            radii=cfg["radii"],
            num_neighbors=cfg["num_neighbors"],
        )

    raise ValueError(f"Unknown global encoder type: {encoder_type}")


def build_pc_patch_tokenizer(
    tokenizer_type: str,
    pc_dim: int,
    config: Optional[Dict] = None,
) -> nn.Module:
    if tokenizer_type not in PATCH_TOKENIZER_CONFIGS:
        raise ValueError(
            f"Unknown patch tokenizer type: {tokenizer_type}. "
            f"Available types: {sorted(PATCH_TOKENIZER_CONFIGS.keys())}"
        )

    cfg = _merge_config(PATCH_TOKENIZER_CONFIGS[tokenizer_type], config)

    if tokenizer_type == "pointpn":
        return PointPNTokenizer(
            input_channels=pc_dim,
            input_points=cfg["input_points"],
            num_stages=cfg["num_stages"],
            embed_channels=cfg["embed_channels"],
            stage_num_neighbors=cfg["stage_num_neighbors"],
            stage_lga_blocks=cfg["stage_lga_blocks"],
            stage_channel_expansion=cfg["stage_channel_expansion"],
            point_cloud_type=cfg["point_cloud_type"],
        )

    if tokenizer_type == "pointnext_tokenizer":
        return PointNextPatchTokenizer(
            input_channels=pc_dim,
            stem_channels=cfg["stem_channels"],
            token_channels=cfg["token_channels"],
            num_patches=cfg["num_patches"],
            patch_radii=cfg["patch_radii"],
            patch_neighbors=cfg["patch_neighbors"],
        )

    raise ValueError(f"Unknown patch tokenizer type: {tokenizer_type}")