from typing import Dict, Optional

import torch.nn as nn

from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import MultiStagePointNet, PointNet
from dexmani_policy.agents.obs_encoder.pointcloud.pointnet_dense import PointNetDense
from dexmani_policy.agents.obs_encoder.pointcloud.pointnext import PointNextEncoder
from dexmani_policy.agents.obs_encoder.pointcloud.pointnext_tokenizer import PointNextPatchTokenizer

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
    "pointnext_tokenizer": {
        "stem_channels": 64,
        "token_channels": 128,
        "num_patches": 96,
        "patch_radii": (0.04, 0.08),
        "patch_neighbors": (16, 32),
        # ── patch self-attention (disabled by default) ──
        "use_patch_self_attn": False,
        "patch_attn_layers": 4,
        "patch_attn_heads": 4,
        "patch_attn_dropout": 0.0,
        "prepend_global_in_attn": True,
    },
    "pointnet_dense": {
        "out_channels": 128,
        "num_points": 256,
    },
}


def merge_config(default_cfg: Dict, config: Optional[Dict] = None) -> Dict:
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

    cfg = merge_config(GLOBAL_ENCODER_CONFIGS[encoder_type], config)
    if encoder_type == "dp3":
        allowed = {"output_channels"}
        cls = PointNet
    elif encoder_type == "idp3":
        allowed = {"output_channels", "hidden_channels", "num_layers"}
        cls = MultiStagePointNet
    else:
        allowed = set(GLOBAL_ENCODER_CONFIGS["pointnext"]) | {
            "fps_random_config", "sa_layers", "expansion", "use_residual"
        }
        cls = PointNextEncoder
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"{encoder_type}: unsupported configuration keys {sorted(unknown)}")
    return cls(input_channels=pc_dim, **cfg)


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

    cfg = merge_config(PATCH_TOKENIZER_CONFIGS[tokenizer_type], config)
    if tokenizer_type == "pointnext_tokenizer":
        allowed = set(PATCH_TOKENIZER_CONFIGS[tokenizer_type]) | {
            "fps_random_config", "include_global_token"
        }
        cls = PointNextPatchTokenizer
    else:
        allowed = set(PATCH_TOKENIZER_CONFIGS[tokenizer_type])
        cls = PointNetDense
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"{tokenizer_type}: unsupported configuration keys {sorted(unknown)}")
    return cls(input_channels=pc_dim, **cfg)
