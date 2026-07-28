"""Shared ViT encoder base class for DINO, CLIP, and SigLIP backbones.

Provides ``backproject()``, ``patch_tokens_to_featmap()``, ``forward()``,
and ``set_tune_mode()`` with a ``_get_lora_target_modules()`` hook.
Subclasses only supply model-specific config/backbone loading,
``get_global_token()``, and LoRA target module names.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Sequence

from dexmani_policy.agents.obs_encoder.rgb.geometry_processor import GeometryProcessor
from dexmani_policy.agents.obs_encoder.rgb.utils import (
    flatten_batch,
    restore_batch,
    get_patch_grid_size,
    reshape_patch_tokens_to_map,
)


class ViTEncoder(nn.Module):
    """Base class for ViT-based RGB encoders (DINO, CLIP, SigLIP).

    Subclasses must set ``self.backbone``, ``self.patch_size``,
    ``self.hidden_dim``, ``self.num_prefix_tokens``, ``self.out_dim``,
    and ``self.proj`` before calling ``super().__init__()``.

    Subclasses must implement:
      - ``get_global_token(outputs, patch_tokens)``
      - ``_get_lora_target_modules()``
    """

    def __init__(self):
        super().__init__()
        self.geometry_processor = GeometryProcessor()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _extract_patch_tokens(self, outputs) -> torch.Tensor:
        """Extract patch tokens from backbone output, skipping prefix tokens.

        SigLIP overrides this because its ``last_hidden_state`` does not
        include a CLS token (``num_prefix_tokens`` = 0).
        """
        return outputs.last_hidden_state[:, self.num_prefix_tokens:]

    def _get_lora_target_modules(self) -> list[str]:
        """Return the LoRA target module names for this backbone variant."""
        raise NotImplementedError("Subclass must implement _get_lora_target_modules()")

    # ------------------------------------------------------------------
    # Shared methods (previously duplicated in dino/clip/siglip)
    # ------------------------------------------------------------------

    def set_tune_mode(self, tune_mode: str) -> None:
        self.tune_mode = tune_mode

        if tune_mode == "freeze":
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            return

        if tune_mode == "full":
            self.backbone.requires_grad_(True)
            return

        if tune_mode == "lora":
            from peft import LoraConfig, get_peft_model

            self.backbone.requires_grad_(False)

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=self._get_lora_target_modules(),
                bias="none",
                use_rslora=True,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

            # Match LoRA dtype to backbone dtype (bfloat16).
            backbone_dtype = next(self.backbone.parameters()).dtype
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.data = param.data.to(dtype=backbone_dtype)
            return

        raise ValueError(f"Unsupported tune_mode: {tune_mode}")

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.ndim < 4 or rgb.shape[-3] != 3:
            raise ValueError(f"rgb should have shape [..., 3, H, W], got {tuple(rgb.shape)}")

        if self.tune_mode == "freeze":
            self.backbone.eval()

        flat_rgb, leading_shape = flatten_batch(rgb, trailing_ndim=3)
        outputs = self.backbone(pixel_values=flat_rgb, return_dict=True)

        patch_tokens = self._extract_patch_tokens(outputs)
        patch_tokens = self.proj(patch_tokens)
        global_token = self.get_global_token(outputs, patch_tokens)

        return {
            "patch_tokens": restore_batch(patch_tokens, leading_shape),
            "global_token": restore_batch(global_token, leading_shape),
        }

    def backproject(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: Optional[torch.Tensor] = None,
        depth_scale: float = 1000.0,
        min_depth: float = 0.0,
        max_depth: Optional[float] = None,
    ) -> Dict[str, object]:
        dense_geometry = self.geometry_processor.backproject_depth(
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        patch_geometry = self.geometry_processor.pool_patch_coordinates(
            coords=dense_geometry["coords"],
            valid_mask=dense_geometry["valid_mask"],
            patch_size=self.patch_size,
        )

        patch_coords = patch_geometry["patch_coords"]
        return {
            "patch_coords": patch_coords,
            "patch_valid_mask": patch_geometry["patch_valid_mask"],
            "geometry_meta": {
                "coord_frame": dense_geometry["coord_frame"],
                "depth_scale": dense_geometry["depth_scale"],
                "min_depth": dense_geometry["min_depth"],
                "max_depth": dense_geometry["max_depth"],
                "patch_grid_size": patch_geometry["patch_grid_size"],
                "patch_hw": patch_geometry["patch_hw"],
                "leading_shape": tuple(patch_coords.shape[:-2]),
            },
        }

    def patch_tokens_to_featmap(self, patch_tokens: torch.Tensor, image_hw: Sequence[int]) -> torch.Tensor:
        patch_grid_size = get_patch_grid_size((int(image_hw[0]), int(image_hw[1])), self.patch_size)
        flat_patch_tokens, leading_shape = flatten_batch(patch_tokens, trailing_ndim=2)
        feature_map = reshape_patch_tokens_to_map(flat_patch_tokens, patch_grid_size)
        return restore_batch(feature_map, leading_shape)
