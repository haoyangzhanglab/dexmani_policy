import torch.nn as nn
from typing import Dict, Optional

from dexmani_policy.agents.obs_encoder.pointcloud import(
    PointNet,
    MultiStagePointNet,
    PointNextEncoder,
    PointPNTokenizer,
    PointNextPatchTokenizer,
)


GLOBAL_ENCODER_CONFIGS: Dict[str, Dict] = {
    "dp3": {
        "pc_out_dim": 256,
    },
    "idp3": {
        "pc_out_dim": 256,
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
        "num_points": 1024,
        "num_stages": 3,
        "embed_dim": 64,
        "k_neighbors": (24, 24, 16),
        "lga_blocks": (2, 2, 1),
        "dim_expansion": (2, 2, 2),
        "point_cloud_type": "scan",
    },
    "tokenizer": {
        "stem_channels": 64,
        "token_channels": 128,
        "num_patches": 96,
        "patch_radii": (0.04, 0.08),
        "patch_neighbors": (16, 32),
        "global_radius": 0.16,
        "global_neighbors": 32,
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
            in_channels=pc_dim,
            out_channels=cfg["pc_out_dim"],
        )

    if encoder_type == "idp3":
        return MultiStagePointNet(
            in_channels=pc_dim,
            out_channels=cfg["pc_out_dim"],
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
            in_channels=pc_dim,
            input_points=cfg["num_points"],
            num_stages=cfg["num_stages"],
            embed_dim=cfg["embed_dim"],
            k_neighbors=cfg["k_neighbors"],
            lga_blocks=cfg["lga_blocks"],
            dim_expansion=cfg["dim_expansion"],
            point_cloud_type=cfg["point_cloud_type"],
        )

    if tokenizer_type == "tokenizer":
        return PointNextPatchTokenizer(
            input_channels=pc_dim,
            stem_channels=cfg["stem_channels"],
            token_channels=cfg["token_channels"],
            num_patches=cfg["num_patches"],
            patch_radii=cfg["patch_radii"],
            patch_neighbors=cfg["patch_neighbors"],
            global_radius=cfg["global_radius"],
            global_neighbors=cfg["global_neighbors"],
        )

    raise ValueError(f"Unknown patch tokenizer type: {tokenizer_type}")