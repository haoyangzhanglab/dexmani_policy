"""Shared type aliases for RGB backbone configuration.

These are the canonical definitions; individual backbones use subsets appropriate
to their architecture (e.g. ResNet does not support ``"lora"`` tune mode).
"""

from typing import Literal

TuneMode = Literal["freeze", "lora", "full"]
"""How to tune the RGB backbone: freeze all, LoRA fine-tune, or full fine-tune."""

GlobalTokenType = Literal["cls", "avg", "pooler"]
"""How to aggregate patch tokens into a global image representation.

ViT backbones support all three; ResNet uses only ``"avg"``.
"""

NormMode = Literal["batch_norm", "frozen_bn", "group_norm", "identity"]
"""Normalization mode for the backbone feature map.

ResNet supports ``"frozen_bn"`` and ``"group_norm"``; most configs use
``"group_norm"``.
"""
