from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModel

from .lora import apply_lora


class DINOEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        tune_mode: Literal["freeze", "full", "lora"] = "freeze",
        global_token_type: Literal["avg", "cls", "pooler"] = "avg",
        out_dim: int | None = None,
        lora_dtype: Literal["backbone", "float32"] = "backbone",
        architecture: dict | None = None,
        load_pretrained: bool = True,
    ):
        super().__init__()
        if global_token_type not in ("avg", "cls", "pooler"):
            raise ValueError(f"Unsupported global_token_type: {global_token_type}")

        self.model_name = model_name
        self.tune_mode = tune_mode
        self.global_token_type = global_token_type
        if OmegaConf.is_config(architecture):
            architecture = OmegaConf.to_container(architecture, resolve=True)
        config = (
            AutoConfig.for_model(**dict(architecture))
            if architecture is not None
            else AutoConfig.from_pretrained(model_name)
        )
        config._attn_implementation = "sdpa"
        self.architecture = config.to_dict()
        self.backbone = (
            AutoModel.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
            if load_pretrained
            else AutoModel.from_config(config, torch_dtype=torch.bfloat16)
        )

        self.patch_size = int(self.backbone.config.patch_size)
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.num_prefix_tokens = 1 + self.num_register_tokens
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        self.proj = (
            nn.Identity()
            if self.out_dim == self.hidden_dim
            else nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.configure_finetuning(tune_mode, lora_dtype=lora_dtype)

    def configure_finetuning(
        self,
        tune_mode: Literal["freeze", "full", "lora"],
        lora_dtype: Literal["backbone", "float32"] = "backbone",
    ) -> None:
        if tune_mode == "lora":
            ffn_targets = (
                ["weights_in", "weights_out"]
                if getattr(self.backbone.config, "use_swiglu_ffn", False)
                else ["fc1", "fc2"]
            )
            targets = ["query", "key", "value", "dense", *ffn_targets]
            self.backbone = apply_lora(self.backbone, targets, dtype=lora_dtype)
        elif tune_mode in ("freeze", "full"):
            self.backbone.requires_grad_(tune_mode == "full")
        else:
            raise ValueError(f"Unsupported tune_mode: {tune_mode}")

        self.tune_mode = tune_mode
        self.backbone.train(self.training and tune_mode != "freeze")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tune_mode == "freeze":
            self.backbone.eval()
        return self

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.proj, nn.Identity):
            features = features.to(dtype=self.proj.weight.dtype)
        return self.proj(features)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        leading_shape = rgb.shape[:-3]
        flat_rgb = rgb.reshape(-1, *rgb.shape[-3:])
        outputs = self.backbone(pixel_values=flat_rgb, return_dict=True)
        patch_tokens = self.project_features(outputs.last_hidden_state[:, self.num_prefix_tokens :])
        if self.global_token_type == "avg":
            global_token = patch_tokens.mean(dim=1)
        elif self.global_token_type == "cls":
            global_token = self.project_features(outputs.last_hidden_state[:, 0])
        else:
            if outputs.pooler_output is None:
                raise ValueError(f"{self.model_name} does not provide pooler_output")
            global_token = self.project_features(outputs.pooler_output)

        return {
            "patch_tokens": patch_tokens.reshape(*leading_shape, *patch_tokens.shape[1:]),
            "global_token": global_token.reshape(*leading_shape, *global_token.shape[1:]),
        }

    def tokens_to_feature_map(
        self, patch_tokens: torch.Tensor, image_hw: Sequence[int]
    ) -> torch.Tensor:
        height, width = int(image_hw[0]), int(image_hw[1])
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch size {self.patch_size}."
            )
        grid = (height // self.patch_size, width // self.patch_size)
        return patch_tokens.unflatten(-2, grid).movedim(-1, -3).contiguous()
