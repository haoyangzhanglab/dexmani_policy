import torch
import torch.nn as nn
from typing import Dict

from dexmani_policy.agents.common.mlp import create_mlp
from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.obs_encoder.rgb.registry import (
    build_backbone, BackboneName, build_backbone as build_backbone_and_image_processor,
)


class DPEncoder(nn.Module):
    def __init__(
        self,
        rgb_backbone_name: BackboneName = "dino",
        state_dim: int = 19,
    ):
        super().__init__()
        self.modality_keys = ["rgb", "joint_state"]
        self.rgb_backbone_name = rgb_backbone_name
        self.backbone, self.image_processor = build_backbone_and_image_processor(rgb_backbone_name)
        self.state_out_dim = 64
        self.state_mlp = create_mlp(state_dim, [64, self.state_out_dim])


    def forward(self, observations: Dict) -> torch.Tensor:
        for key in self.modality_keys:
            if key not in observations:
                raise KeyError(f"Required modality key '{key}' missed")
            
        rgb = observations["rgb"]
        state = observations["joint_state"]
    
        if rgb.shape[:-3] != state.shape[:-1]:
            raise ValueError(
                f"rgb leading dims {tuple(rgb.shape[:-3])} != "
                f"joint_state leading dims {tuple(state.shape[:-1])}"
            )
        
        rgb = self.image_processor.process_images(rgb).image
        vision_out = self.backbone(rgb)
        image_feat = vision_out["global_token"]
        state_feat = self.state_mlp(state)
        feat = torch.cat([image_feat, state_feat], dim=-1)
        return feat

    @property
    def out_shape(self):
        return self.backbone.out_dim + self.state_out_dim

    def get_optim_groups(self, weight_decay):
        return get_optim_group_with_no_decay(self, weight_decay)



def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, n_obs_steps = 2, 2
    height, width = 480, 640
    state_dim = 19

    observations = {
        "rgb": torch.randint(
            0,
            256,
            (batch_size, n_obs_steps, height, width, 3),
            dtype=torch.uint8,
            device=device,
        ),
        "joint_state": torch.randn(batch_size, n_obs_steps, state_dim, device=device),
    }

    for name in ["resnet", "clip", "dino", "siglip"]:
        encoder = DPEncoder(rgb_backbone_name=name, state_dim=state_dim).to(device)
        encoder.eval()
        with torch.no_grad():
            feat = encoder(observations)

        print(f"=== DPEncoder ({name}) ===")
        print("rgb:", tuple(observations["rgb"].shape), observations["rgb"].dtype)
        print("joint_state:", tuple(observations["joint_state"].shape), observations["joint_state"].dtype)
        print("feat:", tuple(feat.shape), feat.dtype)
        print("out_shape:", encoder.out_shape)
        if name == "resnet":
            trainable = any(param.requires_grad for param in encoder.backbone.backbone.parameters())
            print("resnet_tune_mode:", encoder.backbone.tune_mode)
            print("resnet_trainable:", trainable)


if __name__ == "__main__":
    example()
