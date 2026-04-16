import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet, MultiStagePointNet
from dexmani_policy.agents.obs_encoder.pointcloud.pointnext import PointNextEncoder
from dexmani_policy.agents.obs_encoder.pointcloud.point_pn import PointPNTokenizer
from dexmani_policy.agents.obs_encoder.pointcloud.pointnext_tokenizer import PointNextPatchTokenizer


def build_pc_global_encoder(
    type: str,
    pc_dim: int,
    config: Optional[Dict] = None,
) -> Tuple[nn.Module, int, int]:
    """构建全局点云编码器。返回 (encoder, seq_len=1, out_dim)。"""
    if type == "dp3":
        cfg = {"pc_out_dim": 256}
        if config:
            cfg.update(config)
        encoder = PointNet(in_channels=pc_dim, out_channels=cfg["pc_out_dim"])
        out_dim = cfg["pc_out_dim"]

    elif type == "idp3":
        cfg = {"pc_out_dim": 256}
        if config:
            cfg.update(config)
        encoder = MultiStagePointNet(in_channels=pc_dim, out_channels=cfg["pc_out_dim"])
        out_dim = cfg["pc_out_dim"]

    elif type == "pointnext":
        cfg = {
            "output_channels": 256,
            "stage_depths": (1, 2, 2),
            "stage_strides": (1, 2, 2),
            "stage_channels": (64, 128, 256),
            "radii": (0.04, 0.08, 0.16),
            "num_neighbors": (24, 24, 32),
        }
        if config:
            cfg.update(config)
        encoder = PointNextEncoder(input_channels=pc_dim, **cfg)
        out_dim = cfg["output_channels"]

    else:
        raise ValueError(f"Unknown global encoder type: {type}")

    return encoder, 1, out_dim


PC_TOKENIZER_CONFIGS: Dict[str, Dict] = {
    "pointpn": {
        "num_points": 1024,
        "num_stages": 3,
        "embed_dim": 64,
        "k_neighbors": [24, 24, 16],
        "lga_blocks": [2, 2, 1],
        "dim_expansion": [2, 2, 2],
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


def build_pc_patch_tokenizer(
    type: str,
    pc_dim: int,
    config: Optional[Dict] = None,
) -> Tuple[nn.Module, int, int]:
    """构建点云 patch tokenizer。返回 (tokenizer, seq_len, out_dim)。"""
    cfg = dict(PC_TOKENIZER_CONFIGS[type])
    if config:
        cfg.update(config)

    if type == "pointpn":
        encoder = PointPNTokenizer(
            in_channels=pc_dim,
            input_points=cfg["num_points"],
            num_stages=cfg["num_stages"],
            embed_dim=cfg["embed_dim"],
            k_neighbors=cfg["k_neighbors"],
            lga_blocks=cfg["lga_blocks"],
            dim_expansion=cfg["dim_expansion"],
            point_cloud_type=cfg["point_cloud_type"],
        )
        seq_len = cfg["num_points"] // (2 ** cfg["num_stages"]) + 1
        out_dim = encoder.out_channels

    elif type == "tokenizer":
        encoder = PointNextPatchTokenizer(
            input_channels=pc_dim,
            stem_channels=cfg["stem_channels"],
            token_channels=cfg["token_channels"],
            num_patches=cfg["num_patches"],
            patch_radii=cfg["patch_radii"],
            patch_neighbors=cfg["patch_neighbors"],
            global_radius=cfg["global_radius"],
            global_neighbors=cfg["global_neighbors"],
        )
        seq_len = cfg["num_patches"] + 1
        out_dim = cfg["token_channels"]

    else:
        raise ValueError(f"Unknown tokenizer type: {type}")

    return encoder, seq_len, out_dim
