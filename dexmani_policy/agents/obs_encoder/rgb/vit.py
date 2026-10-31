from typing import Literal, cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModel, SiglipVisionConfig, SiglipVisionModel

from .lora import apply_lora


class ViTEncoder(nn.Module):
    def __init__(
        self,
        kind: Literal["dino", "siglip"],
        tune_mode: Literal["freeze", "full", "lora"] = "freeze",
        global_token_type: Literal["avg", "cls", "pooler"] = "avg",
        out_dim: int | None = None,
        lora_dtype: Literal["backbone", "float32"] = "backbone",
        architecture: dict | None = None,
    ):
        super().__init__()
        token_types = ("avg", "cls", "pooler") if kind == "dino" else ("avg", "pooler")
        if global_token_type not in token_types:
            raise ValueError(f"Unsupported global_token_type: {global_token_type}")

        self.kind = kind
        self.tune_mode = tune_mode
        self.global_token_type = global_token_type
        if OmegaConf.is_config(architecture):
            architecture = cast(dict, OmegaConf.to_container(architecture, resolve=True))
        if architecture is None:
            raise ValueError("ViTEncoder requires an explicit architecture; resolve it before construction")
        if kind == "dino":
            config = AutoConfig.for_model(**dict(architecture))
        elif kind == "siglip":
            config = SiglipVisionConfig.from_dict(dict(architecture))
            config.vision_use_head = global_token_type == "pooler"
        else:
            raise ValueError(f"Unsupported ViT kind: {kind}")
        config._attn_implementation = "sdpa"
        self.effective_config = dict(
            tune_mode=tune_mode, global_token_type=global_token_type,
            out_dim=out_dim, lora_dtype=lora_dtype, architecture=config.to_dict(),
        )
        if kind == "dino":
            self.backbone = AutoModel.from_config(config, torch_dtype=torch.bfloat16)
        else:
            self.backbone = SiglipVisionModel(config).to(dtype=torch.bfloat16)

        self.patch_size = int(self.backbone.config.patch_size)
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.num_prefix_tokens = (
            1 + int(getattr(config, "num_register_tokens", 0)) if kind == "dino" else 0
        )
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        self.proj = (
            nn.Identity()
            if self.out_dim == self.hidden_dim
            else nn.Linear(self.hidden_dim, self.out_dim)
        )
        if tune_mode == "lora":
            if kind == "dino":
                ffn_targets = (
                    ["weights_in", "weights_out"]
                    if getattr(self.backbone.config, "use_swiglu_ffn", False)
                    else ["fc1", "fc2"]
                )
                targets = ["query", "key", "value", "dense", *ffn_targets]
            else:
                targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            self.backbone = apply_lora(self.backbone, targets, dtype=lora_dtype)
        elif tune_mode in ("freeze", "full"):
            self.backbone.requires_grad_(tune_mode == "full")
        else:
            raise ValueError(f"Unsupported tune_mode: {tune_mode}")

        self.backbone.train(self.training and tune_mode != "freeze")

    def initialize_weights(self, source: str | None) -> None:
        if source is None:
            return
        backbone = self.backbone.get_base_model() if self.tune_mode == "lora" else self.backbone
        model_class = AutoModel if self.kind == "dino" else SiglipVisionModel
        pretrained = model_class.from_pretrained(source, config=backbone.config, torch_dtype=torch.bfloat16)
        source_state = pretrained.state_dict()
        state = backbone.state_dict()
        # PEFT inserts base_layer around adapted Linear modules; adapters stay freshly initialized.
        base_keys = {key: key.replace(".base_layer.", ".") for key in state if ".lora_" not in key}
        if set(base_keys.values()) != set(source_state):
            raise ValueError("Pretrained RGB parameters do not match the configured architecture")
        state.update({key: source_state[source_key] for key, source_key in base_keys.items()})
        backbone.load_state_dict(state, strict=True)

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
        outputs = self.backbone(pixel_values=rgb, return_dict=True)
        patch_tokens = self.project_features(outputs.last_hidden_state[:, self.num_prefix_tokens :])
        if self.global_token_type == "avg":
            global_token = patch_tokens.mean(dim=1)
        elif self.global_token_type == "cls":
            global_token = self.project_features(outputs.last_hidden_state[:, 0])
        else:
            if outputs.pooler_output is None:
                raise ValueError(f"{self.kind} does not provide pooler_output")
            global_token = self.project_features(outputs.pooler_output)

        return {
            "patch_tokens": patch_tokens,
            "global_token": global_token,
        }
