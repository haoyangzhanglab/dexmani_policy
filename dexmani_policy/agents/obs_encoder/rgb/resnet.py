from typing import Literal

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import FrozenBatchNorm2d


def convert_group_norm(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_groups = min(32, child.num_features)
            while child.num_features % num_groups != 0:
                num_groups -= 1
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features),
            )
        else:
            convert_group_norm(child)
    return module


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        tune_mode: Literal["freeze", "full"] = "full",
        norm_mode: Literal["batch_norm", "frozen_bn", "group_norm"] = "group_norm",
        out_dim: int | None = None,
        weights=None,
        load_pretrained: bool = True,
    ):
        super().__init__()
        if norm_mode not in ("batch_norm", "frozen_bn", "group_norm"):
            raise ValueError(
                f"Unsupported norm_mode: {norm_mode!r}; expected batch_norm, frozen_bn, or group_norm"
            )
        if (
            load_pretrained
            and weights is not None
            and tune_mode == "freeze"
            and norm_mode == "group_norm"
        ):
            raise ValueError(
                "Frozen pretrained ResNet requires batch_norm/frozen_bn; "
                "group_norm is a separate full-finetuning recipe"
            )

        self.effective_config = dict(
            model_name=model_name, tune_mode=tune_mode, norm_mode=norm_mode,
            out_dim=out_dim,
            weights=weights, load_pretrained=load_pretrained,
        )
        self.tune_mode = tune_mode

        norm_layer = FrozenBatchNorm2d if norm_mode == "frozen_bn" else nn.BatchNorm2d
        backbone = getattr(torchvision.models, model_name)(
            weights=weights if load_pretrained else None, norm_layer=norm_layer
        )

        if norm_mode == "group_norm":
            backbone = convert_group_norm(backbone)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.hidden_dim = int(backbone.fc.in_features)
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        self.proj = (
            nn.Identity()
            if self.out_dim == self.hidden_dim
            else nn.Conv2d(
                self.hidden_dim,
                self.out_dim,
                kernel_size=1,
            )
        )

        if tune_mode not in ("freeze", "full"):
            raise ValueError(f"Unsupported tune_mode: {tune_mode}")
        self.backbone.requires_grad_(tune_mode == "full")
        self.backbone.train(self.training and tune_mode != "freeze")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tune_mode == "freeze":
            self.backbone.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        leading_shape = rgb.shape[:-3]
        flat_rgb = rgb.reshape(-1, *rgb.shape[-3:]).to(memory_format=torch.channels_last)
        feature_map = self.backbone(flat_rgb)
        feature_map = self.proj(feature_map)

        patch_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        global_token = feature_map.mean(dim=(-2, -1))

        return {
            "patch_tokens": patch_tokens.reshape(*leading_shape, *patch_tokens.shape[1:]),
            "global_token": global_token.reshape(*leading_shape, *global_token.shape[1:]),
        }
