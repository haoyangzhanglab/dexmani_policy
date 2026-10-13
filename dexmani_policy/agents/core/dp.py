import torch
import torch.nn as nn

from dexmani_policy.agents.core.base import UNetDiffusionAgent
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.agents.obs_encoder.rgb.r3m import R3M
from dexmani_policy.agents.obs_encoder.rgb.registry import build_backbone
from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNet


class DPObsEncoder(nn.Module):
    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        """Return the exact raw observation fields consumed by ``forward``."""
        return ("joint_state", "rgb")

    def __init__(
        self,
        rgb_backbone_name: str,
        state_dim: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        rgb_backbone_config: dict = None,
    ):
        super().__init__()
        cfg = dict(rgb_backbone_config or {})
        if "crop_ratio" in cfg:
            raise ValueError(
                "rgb_backbone_config.crop_ratio is no longer supported; "
                "configure Dataset rgb_random_crop_size instead."
            )
        self.backbone, self.image_processor = build_backbone(rgb_backbone_name, config=cfg)
        self.state_mlp = create_state_mlp(state_dim, state_out_dim)
        self.n_obs_steps = n_obs_steps
        self.out_dim = self.backbone.out_dim + self.state_mlp.out_dim

    def forward(self, obs: dict):
        rgb = obs["rgb"]  # uint8 or float32 [0,1], from Dataset/saved eval preprocessing
        rgb = self.image_processor.process_images(rgb, validate_float_range=False)["image"]

        # Use channels-last memory format for CNN backbones; shape stays NCHW.
        if isinstance(self.backbone, (ResNet, R3M)):
            rgb = rgb.to(memory_format=torch.channels_last)

        feat = torch.cat(
            [
                self.backbone(rgb)["global_token"],
                self.state_mlp(obs["joint_state"]),
            ],
            dim=-1,
        )
        B = feat.shape[0] // self.n_obs_steps
        return feat.reshape(B, -1), {}


class DPAgent(UNetDiffusionAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        rgb_backbone_name: str,
        state_dim: int,
        state_out_dim: int = 64,
        rgb_backbone_config: dict = None,
        **kwargs,
    ):
        obs_encoder = DPObsEncoder(
            rgb_backbone_name,
            state_dim,
            n_obs_steps,
            state_out_dim,
            rgb_backbone_config=rgb_backbone_config,
        )
        super().__init__(obs_encoder, horizon, n_obs_steps, n_action_steps, action_dim, **kwargs)
