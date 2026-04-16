import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Tuple

from dexmani_policy.agents.obs_encoder.dp3 import DP3Encoder
from dexmani_policy.agents.obs_encoder.pointcloud.point_pn import PointPNEncoder
from dexmani_policy.agents.obs_encoder.pointcloud.tokenizer import PointNextPatchTokenizer

PC_ENCODER_CONFIGS: Dict[str, Dict] = {
    "idp3": {
        "pc_out_dim": 128,
        "num_points": 1024,
    },
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


def build_idp3(
    pc_dim: int, config: Dict
) -> Tuple[nn.Module, int, int, Callable]:
    encoder = DP3Encoder(
        type="idp3",
        pc_dim=pc_dim,
        pc_out_dim=config["pc_out_dim"],
        point_wise=True,
    )
    seq_len = config["num_points"] + 1  # +1 for global_token
    out_dim = config["pc_out_dim"]

    def extract_fn(pc: torch.Tensor) -> torch.Tensor:
        patch_token, patch_center, global_token = encoder(pc)
        return torch.cat([global_token, patch_token], dim=1)

    return encoder, seq_len, out_dim, extract_fn


def build_pointpn(
    pc_dim: int, config: Dict
) -> Tuple[nn.Module, int, int, Callable]:
    encoder = PointPNEncoder(
        in_channels=pc_dim,
        input_points=config["num_points"],
        num_stages=config["num_stages"],
        embed_dim=config["embed_dim"],
        k_neighbors=config["k_neighbors"],
        lga_blocks=config["lga_blocks"],
        dim_expansion=config["dim_expansion"],
        point_cloud_type=config["point_cloud_type"],
    )
    seq_len = config["num_points"] // (2 ** config["num_stages"]) + 1  # +1 for global_token
    out_dim = encoder.out_channels

    def extract_fn(pc: torch.Tensor) -> torch.Tensor:
        patch_token, patch_center, global_token = encoder(pc)
        return torch.cat([global_token, patch_token], dim=1)

    return encoder, seq_len, out_dim, extract_fn


def build_tokenizer(
    pc_dim: int, config: Dict
) -> Tuple[nn.Module, int, int, Callable]:
    encoder = PointNextPatchTokenizer(
        input_channels=pc_dim,
        stem_channels=config["stem_channels"],
        token_channels=config["token_channels"],
        num_patches=config["num_patches"],
        patch_radii=config["patch_radii"],
        patch_neighbors=config["patch_neighbors"],
        global_radius=config["global_radius"],
        global_neighbors=config["global_neighbors"],
    )
    seq_len = config["num_patches"] + 1
    out_dim = config["token_channels"]

    def extract_fn(pc: torch.Tensor) -> torch.Tensor:
        patch_token, patch_center, global_token = encoder(pc)
        return torch.cat([global_token, patch_token], dim=1)

    return encoder, seq_len, out_dim, extract_fn


PC_BUILDERS: Dict[str, Callable] = {
    "idp3": build_idp3,
    "pointpn": build_pointpn,
    "tokenizer": build_tokenizer,
}


def build_pc_encoder(
    type: str,
    pc_dim: int,
    config: Optional[Dict] = None,
) -> Tuple[nn.Module, int, int, Callable]:
    """Build a point cloud encoder.

    Returns:
        (encoder, seq_len, out_dim, extract_fn)

    所有编码器统一返回 (patch_token, patch_center, global_token) 三 tuple，
    语义与 PointNextPatchTokenizer 一致。
    """
    cfg = dict(PC_ENCODER_CONFIGS[type])
    if config:
        cfg.update(config)
    return PC_BUILDERS[type](pc_dim, cfg)
