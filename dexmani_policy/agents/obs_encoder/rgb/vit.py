from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, SiglipVisionConfig, SiglipVisionModel


class ViTEncoder(nn.Module):
    def __init__(
        self,
        kind: Literal["dino", "dinov3", "siglip"],
        tune_mode: Literal["freeze", "full", "lora"] = "freeze",
        global_token_type: Literal["avg", "cls", "pooler"] = "avg",
        out_dim: int | None = None,
        architecture: dict | None = None,
    ):
        super().__init__()
        token_types = ("avg", "cls", "pooler") if kind == "dino" else ("avg", "pooler")
        if kind == "dinov3":
            token_types = ("avg", "cls")
        if global_token_type not in token_types:
            raise ValueError(f"Unsupported global_token_type: {global_token_type}")

        self.kind = kind
        self.tune_mode = tune_mode
        self.global_token_type = global_token_type
        if kind == "dinov3":
            import timm

            architecture = {"model_name": "vit_small_patch16_dinov3", **(architecture or {})}
            self.backbone = timm.create_model(**architecture, pretrained=False)
            self.patch_size = self.backbone.patch_embed.patch_size[0]
            self.hidden_dim = self.backbone.num_features
            self.num_prefix_tokens = self.backbone.num_prefix_tokens
        else:
            if architecture is None:
                raise ValueError("HF ViT requires an explicit architecture; resolve it before construction")
            if kind == "dino":
                config = AutoConfig.for_model(**architecture)
            elif kind == "siglip":
                # HF from_dict mutates its input; preserve the caller's architecture.
                config = SiglipVisionConfig.from_dict(dict(architecture))
                config.vision_use_head = global_token_type == "pooler"
            else:
                raise ValueError(f"Unsupported ViT kind: {kind}")
            config._attn_implementation = "sdpa"
            architecture = config.to_dict()
            if kind == "dino":
                self.backbone = AutoModel.from_config(config, torch_dtype=torch.float32)
            else:
                self.backbone = SiglipVisionModel(config)
            self.patch_size = int(config.patch_size)
            self.hidden_dim = int(config.hidden_size)
            self.num_prefix_tokens = (
                1 + int(getattr(config, "num_register_tokens", 0)) if kind == "dino" else 0
            )
        self.effective_config = dict(
            tune_mode=tune_mode, global_token_type=global_token_type,
            out_dim=out_dim, architecture=architecture,
        )
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        self.proj = (
            nn.Identity()
            if self.out_dim == self.hidden_dim
            else nn.Linear(self.hidden_dim, self.out_dim)
        )
        if tune_mode == "lora":
            from peft import LoraConfig, get_peft_model

            if kind == "dino":
                ffn_targets = (
                    ["weights_in", "weights_out"]
                    if getattr(self.backbone.config, "use_swiglu_ffn", False)
                    else ["fc1", "fc2"]
                )
                targets = ["query", "key", "value", "dense", *ffn_targets]
            elif kind == "dinov3":
                targets = [
                    name for name, module in self.backbone.named_modules()
                    if name.startswith("blocks.") and isinstance(module, nn.Linear)
                ]
            else:
                targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=targets, bias="none", use_rslora=True,
            ))
        elif tune_mode in ("freeze", "full"):
            self.backbone.requires_grad_(tune_mode == "full")
        else:
            raise ValueError(f"Unsupported tune_mode: {tune_mode}")

        self.backbone.train(self.training and tune_mode != "freeze")

    def initialize_weights(self, source: str | None) -> None:
        if source is None:
            return
        backbone = self.backbone.get_base_model() if self.tune_mode == "lora" else self.backbone
        if self.kind == "dinov3":
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            path = Path(source).expanduser()
            weights = (
                path / "model.safetensors" if path.is_dir()
                else hf_hub_download(source, "model.safetensors")
            )
            source_state = load_file(str(weights))
        else:
            model_class = AutoModel if self.kind == "dino" else SiglipVisionModel
            pretrained = model_class.from_pretrained(source, config=backbone.config, torch_dtype=torch.float32)
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

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.kind == "dinov3":
            tokens = self.backbone.forward_features(rgb)
            pooled = None
        else:
            outputs = self.backbone(pixel_values=rgb, return_dict=True)
            tokens, pooled = outputs.last_hidden_state, outputs.pooler_output
        patch_tokens = self.proj(tokens[:, self.num_prefix_tokens :])
        if self.global_token_type == "avg":
            global_token = patch_tokens.mean(dim=1)
        elif self.global_token_type == "cls":
            global_token = self.proj(tokens[:, 0])
        else:
            if pooled is None:
                raise ValueError(f"{self.kind} does not provide pooler_output")
            global_token = self.proj(pooled)

        return {
            "patch_tokens": patch_tokens,
            "global_token": global_token,
        }
