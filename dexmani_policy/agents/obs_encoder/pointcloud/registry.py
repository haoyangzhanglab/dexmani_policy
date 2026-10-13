from typing import Dict, Optional

import torch.nn as nn

from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import MultiStagePointNet, PointNet
from dexmani_policy.agents.obs_encoder.pointcloud.pointnext import PointNextEncoder

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

    cfg = dict(GLOBAL_ENCODER_CONFIGS[encoder_type])
    cfg.update(config or {})
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
