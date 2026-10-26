"""RGB global features and ordered joint-state history shared by RGB policies."""

import torch
from torch import nn

from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.obs_encoder.rgb.build import build_rgb_encoder
from dexmani_policy.agents.obs_encoder.rgb.resnet import ResNetEncoder


class RGBStateObservationEncoder(nn.Module):
    consumed_observation_fields = ("joint_state", "rgb")

    def __init__(
        self,
        rgb_backbone_name: str,
        state_dim: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        rgb_backbone_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone, self.image_processor = build_rgb_encoder(
            rgb_backbone_name, config=rgb_backbone_config
        )
        self.state_mlp = StateEncoder(state_dim, state_out_dim)
        self.n_obs_steps = n_obs_steps
        self.out_dim = self.backbone.out_dim + self.state_mlp.out_dim

    @staticmethod
    def prepare_restore_config(agent):
        from omegaconf import DictConfig

        rgb = dict(agent.rgb_backbone_config)
        if agent.rgb_backbone_name == "resnet" and not {"tune_mode", "norm_mode"} <= rgb.keys():
            raise ValueError("Saved ResNet requires explicit tune_mode and norm_mode")
        if agent.rgb_backbone_name in {"dino", "siglip"} and not isinstance(
            rgb.get("architecture"), (dict, DictConfig)
        ):
            raise ValueError("Saved HF backbone requires an architecture mapping")
        rgb["load_pretrained"] = False
        agent.rgb_backbone_config = rgb

    @property
    def effective_config(self):
        return {"rgb_backbone_config": self.backbone.effective_config}

    def forward(self, obs: dict) -> torch.Tensor:
        rgb = self.image_processor.process_images(obs["rgb"])

        if isinstance(self.backbone, ResNetEncoder):
            rgb = rgb.to(memory_format=torch.channels_last)

        features = torch.cat(
            [
                self.backbone(rgb)["global_token"],
                self.state_mlp(obs["joint_state"]),
            ],
            dim=-1,
        )
        return features.unflatten(0, (-1, self.n_obs_steps)).flatten(1)

