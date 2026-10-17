from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn


def apply_lora(
    backbone: nn.Module,
    target_modules: Sequence[str],
    dtype: Literal["backbone", "float32"] = "backbone",
) -> nn.Module:
    """Apply rank-stabilized LoRA with an explicit adapter dtype."""
    from peft import LoraConfig, get_peft_model

    if dtype not in ("backbone", "float32"):
        raise ValueError(f"Unsupported lora_dtype: {dtype!r}")

    backbone.requires_grad_(False)
    backbone_dtype = next(backbone.parameters()).dtype
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=list(target_modules),
        bias="none",
        use_rslora=True,
    )
    backbone = get_peft_model(backbone, config)
    adapter_dtype = torch.float32 if dtype == "float32" else backbone_dtype
    for name, parameter in backbone.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(dtype=adapter_dtype)
    return backbone
