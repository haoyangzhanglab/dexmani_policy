import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from typing import Dict, Literal, Optional, Tuple
from dexmani_policy.agents.obs_encoder.rgb.geometry_processor import WorldCoordGenerator

TuneMode = Literal["freeze", "lora", "full"]
GlobalTokenType = Literal["cls", "avg", "pooler"]



class DINO(nn.Module):
    # 中文注释：forward 做语义编码，backproject 做 patch 级几何对齐。
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        tune_mode: TuneMode = "freeze",
        global_token_type: GlobalTokenType = "cls",
        out_dim: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.global_token_type = global_token_type

        self.backbone = AutoModel.from_pretrained(model_name)
        if not hasattr(self.backbone.config, "patch_size"):
            raise ValueError(f"{model_name} does not look like a ViT-style DINO model.")

        self.patch_size = int(self.backbone.config.patch_size)
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        self.proj = nn.Identity() if self.out_dim == self.hidden_dim else nn.Linear(self.hidden_dim, self.out_dim)
        self.geometry = WorldCoordGenerator()

        self.set_tune_mode(
            tune_mode=tune_mode,
            lora_rank=lora_rank,
            lora_alpha=lora_rank if lora_alpha is None else lora_alpha,
            lora_dropout=lora_dropout,
        )

    def set_tune_mode(
        self,
        tune_mode: TuneMode,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        if tune_mode == "freeze":
            self.backbone.requires_grad_(False)
            return

        if tune_mode == "full":
            self.backbone.requires_grad_(True)
            return

        if tune_mode == "lora":
            self.backbone.requires_grad_(False)
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                bias="none",
                use_rslora=True,
            )
            self.backbone = get_peft_model(self.backbone, config)
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.data = param.data.float()
            return
        raise ValueError(f"Unsupported tune_mode: {tune_mode}")


    def flatten_batch(self, x: torch.Tensor, channel_dim: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        if x.ndim < 4 or x.shape[-3] != channel_dim:
            raise ValueError(f"Expected shape [..., {channel_dim}, H, W].")
        leading_shape = tuple(x.shape[:-3])
        x = x.reshape(-1, *x.shape[-3:])
        return x, leading_shape


    def restore_batch(self, x: torch.Tensor, leading_shape: Tuple[int, ...]) -> torch.Tensor:
        return x.reshape(*leading_shape, *x.shape[1:])


    def get_patch_grid_size(self, image_hw: Tuple[int, int]) -> Tuple[int, int]:
        image_h, image_w = image_hw
        if image_h % self.patch_size != 0 or image_w % self.patch_size != 0:
            raise ValueError(
                f"Input size {(image_h, image_w)} must be divisible by patch size {self.patch_size}."
            )
        return image_h // self.patch_size, image_w // self.patch_size


    def split_patch_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        patch_tokens = hidden_states[:, 1 + self.num_register_tokens:]
        return self.proj(patch_tokens)


    def get_global_token(self, outputs, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.global_token_type == "avg":
            return patch_tokens.mean(dim=1)

        if self.global_token_type == "pooler":
            pooler_output = getattr(outputs, "pooler_output", None)
            if pooler_output is not None:
                return self.proj(pooler_output)

        return self.proj(outputs.last_hidden_state[:, 0])


    def build_feature_map(self, patch_tokens: torch.Tensor, patch_grid_size: Tuple[int, int]) -> torch.Tensor:
        batch_size, token_num, channel_dim = patch_tokens.shape
        grid_h, grid_w = patch_grid_size
        if token_num != grid_h * grid_w:
            raise ValueError(
                f"Patch token number {token_num} does not match patch grid size {patch_grid_size}."
            )
        feature_map = patch_tokens.view(batch_size, grid_h, grid_w, channel_dim)
        return feature_map.permute(0, 3, 1, 2).contiguous()


    def pool_patch_coords(
        self,
        image_world_coords: torch.Tensor,
        image_valid_mask: torch.Tensor,
        patch_grid_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if image_world_coords.ndim != 4 or image_world_coords.shape[1] != 3:
            raise ValueError("image_world_coords should have shape [B, 3, H, W].")
        if image_valid_mask.ndim != 4 or image_valid_mask.shape[1] != 1:
            raise ValueError("image_valid_mask should have shape [B, 1, H, W].")

        grid_h, grid_w = patch_grid_size
        image_h, image_w = image_world_coords.shape[-2:]
        if image_h % grid_h != 0 or image_w % grid_w != 0:
            raise ValueError(
                f"Image size {(image_h, image_w)} is not divisible by patch grid size {patch_grid_size}."
            )

        kernel_h = image_h // grid_h
        kernel_w = image_w // grid_w

        coords = image_world_coords.float()
        mask = image_valid_mask.float()

        coord_num = F.avg_pool2d(
            coords * mask,
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
        )
        coord_den = F.avg_pool2d(
            mask,
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
        )

        coord_map = coord_num / coord_den.clamp_min(1e-6)
        coord_valid_map = coord_den > 1e-6
        patch_coords = coord_map.flatten(2).transpose(1, 2).contiguous()
        patch_valid_mask = coord_valid_map.flatten(2).transpose(1, 2).contiguous()

        return {
            "patch_coords": patch_coords,
            "patch_valid_mask": patch_valid_mask,
            "coord_map": coord_map,
            "coord_valid_map": coord_valid_map,
        }


    def forward(self, rgb: torch.Tensor, return_feature_map: bool = False) -> Dict[str, torch.Tensor]:
        """
        rgb: [..., 3, H, W], 已经过 resize / crop / normalize。
        """
        rgb, leading_shape = self.flatten_batch(rgb, channel_dim=3)
        patch_grid_size = self.get_patch_grid_size(rgb.shape[-2:])

        outputs = self.backbone(pixel_values=rgb, return_dict=True)
        patch_tokens = self.split_patch_tokens(outputs.last_hidden_state)
        global_token = self.get_global_token(outputs, patch_tokens)

        out = {
            "patch_tokens": self.restore_batch(patch_tokens, leading_shape),
            "global_token": self.restore_batch(global_token, leading_shape),
        }

        if return_feature_map:
            feature_map = self.build_feature_map(patch_tokens, patch_grid_size)
            out["feature_map"] = self.restore_batch(feature_map, leading_shape)

        return out


    def backproject(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: Optional[torch.Tensor] = None,
        depth_scale: float = 1000.0,
        min_depth: float = 0.0,
        max_depth: Optional[float] = None,
        return_coord_map: bool = False,
        return_dense: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        depth / intrinsics / camera_to_world 应和 forward 中的 rgb 使用同一套 resize / crop 结果。
        """
        geo_out = self.geometry.compute(
            depths=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            collapse_repeated_camera=True,
        )

        image_world_coords = geo_out["image_world_coords"]
        image_valid_mask = geo_out["image_valid_mask"]

        image_world_coords, leading_shape = self.flatten_batch(image_world_coords, channel_dim=3)
        image_valid_mask, _ = self.flatten_batch(image_valid_mask, channel_dim=1)

        patch_grid_size = self.get_patch_grid_size(image_world_coords.shape[-2:])
        coord_out = self.pool_patch_coords(
            image_world_coords=image_world_coords,
            image_valid_mask=image_valid_mask,
            patch_grid_size=patch_grid_size,
        )

        out = {
            "patch_coords": self.restore_batch(coord_out["patch_coords"], leading_shape),
            "patch_valid_mask": self.restore_batch(coord_out["patch_valid_mask"], leading_shape),
        }

        if return_coord_map:
            out["coord_map"] = self.restore_batch(coord_out["coord_map"], leading_shape)
            out["coord_valid_map"] = self.restore_batch(coord_out["coord_valid_map"], leading_shape)

        if return_dense:
            out["image_world_coords"] = geo_out["image_world_coords"]
            out["image_valid_mask"] = geo_out["image_valid_mask"]

        return out


def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DINO(model_name="facebook/dinov2-small", tune_mode="freeze").to(device)
    encoder.eval()

    rgb = torch.randn(2, 4, 3, 224, 224, device=device)
    depth = torch.randint(1, 2000, (2, 4, 1, 224, 224), device=device).float()
    intrinsics = torch.tensor(
        [[600.0, 0.0, 112.0], [0.0, 600.0, 112.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    camera_to_world = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5]],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        vision_out = encoder(rgb, return_feature_map=True)
        geo_out = encoder.backproject(depth, intrinsics, camera_to_world, return_coord_map=True)

    print("patch_tokens:", tuple(vision_out["patch_tokens"].shape))
    print("global_token:", tuple(vision_out["global_token"].shape))
    print("feature_map:", tuple(vision_out["feature_map"].shape))
    print("patch_coords:", tuple(geo_out["patch_coords"].shape))
    print("patch_valid_mask:", tuple(geo_out["patch_valid_mask"].shape))
    print("coord_map:", tuple(geo_out["coord_map"].shape))


if __name__ == "__main__":
    example()