"""Shared type aliases for RGB backbone configuration.

These are the canonical definitions; individual backbones use subsets appropriate
to their architecture (e.g. ResNet does not support ``"lora"`` tune mode).
"""

from typing import Literal

TuneMode = Literal["freeze", "lora", "full"]
"""How to tune the RGB backbone: freeze all, LoRA fine-tune, or full fine-tune."""

GlobalTokenType = Literal["cls", "avg", "pooler"]
"""How to aggregate patch tokens into a global image representation.

Supported values depend on the backbone: ResNet/R3M use only ``"avg"``;
SigLIP has no CLS token and accepts ``"avg"`` or ``"pooler"``.
"""

NormMode = Literal["batch_norm", "frozen_bn", "group_norm"]
"""Normalization mode for the backbone feature map.

ResNet and R3M support BatchNorm, frozen BatchNorm, and GroupNorm.
"""
