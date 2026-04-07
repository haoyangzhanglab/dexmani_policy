import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection
from typing import Dict, Literal, Optional, Sequence

from dexmani_policy.agents.obs_encoder.rgb.common.image_processor import ImageProcessor
from dexmani_policy.agents.obs_encoder.rgb.common.geometry_processor import GeometryProcessor
from dexmani_policy.agents.obs_encoder.rgb.common.utils import (
    flatten_batch,
    restore_batch,
    get_patch_grid_size,
    reshape_patch_tokens_to_map,
)

TuneMode = Literal["freeze", "lora", "full"]
GlobalTokenType = Literal["cls", "avg", "pooler"]


class CLIP(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        tune_mode: TuneMode = "freeze",
        global_token_type: GlobalTokenType = "pooler",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.tune_mode = tune_mode
        self.global_token_type = global_token_type
        self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)

        if not hasattr(self.backbone.config, "patch_size"):
            raise ValueError(f"{model_name} does not look like a ViT-style CLIP model.")
        if not hasattr(self.backbone.config, "hidden_size"):
            raise ValueError(f"{model_name} is missing hidden_size in backbone config.")
        if not hasattr(self.backbone.config, "projection_dim"):
            raise ValueError(f"{model_name} is missing projection_dim in backbone config.")

        self.patch_size = int(self.backbone.config.patch_size)
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.model_dim = int(self.backbone.config.projection_dim)
        self.num_prefix_tokens = 1
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        self.proj = nn.Identity() if self.out_dim == self.hidden_dim else nn.Linear(self.hidden_dim, self.out_dim)
        self.global_proj = nn.Identity() if self.out_dim == self.model_dim else nn.Linear(self.model_dim, self.out_dim)
        self.geometry_processor = GeometryProcessor()

        self.set_tune_mode(tune_mode)


    def set_tune_mode(self, tune_mode: TuneMode) -> None:
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

            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "visual_projection"]
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=target_modules,
                bias="none",
                use_rslora=True,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.data = param.data.float()
            return

        raise ValueError(f"Unsupported tune_mode: {tune_mode}")


    def get_global_token(self, outputs, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.global_token_type == "avg":
            return patch_tokens.mean(dim=1)

        if self.global_token_type == "pooler":
            image_embeds = getattr(outputs, "image_embeds", None)
            if image_embeds is not None:
                return self.global_proj(image_embeds)

        return self.proj(outputs.last_hidden_state[:, 0])


    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.ndim < 4 or rgb.shape[-3] != 3:
            raise ValueError(f"rgb should have shape [..., 3, H, W], got {tuple(rgb.shape)}")

        if self.tune_mode == "freeze":
            self.backbone.eval()

        flat_rgb, leading_shape = flatten_batch(rgb, trailing_ndim=3)
        outputs = self.backbone(pixel_values=flat_rgb, return_dict=True)

        patch_tokens = outputs.last_hidden_state[:, self.num_prefix_tokens :]
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
    ) -> Dict[str, torch.Tensor]:
        dense_geometry = self.geometry_processor.backproject_depth(
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            collapse_repeated_camera=True,
        )

        patch_geometry = self.geometry_processor.pool_patch_coordinates(
            coords=dense_geometry.coords,
            valid_mask=dense_geometry.valid_mask,
            patch_size=self.patch_size,
        )

        return {
            "patch_coords": patch_geometry.patch_coords,
            "patch_valid_mask": patch_geometry.patch_valid_mask,
        }


    def patch_tokens_to_featmap(self, patch_tokens: torch.Tensor, image_hw: Sequence[int]) -> torch.Tensor:
        patch_grid_size = get_patch_grid_size((int(image_hw[0]), int(image_hw[1])), self.patch_size)
        flat_patch_tokens, leading_shape = flatten_batch(patch_tokens, trailing_ndim=2)
        feature_map = reshape_patch_tokens_to_map(flat_patch_tokens, patch_grid_size)
        return restore_batch(feature_map, leading_shape)



def example() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"

    image_processor = ImageProcessor.from_preset("clip")

    images = torch.randint(0, 256, (16, 2, 480, 640, 3), dtype=torch.uint8)
    depths = torch.randint(1, 2000, (16, 2, 480, 640), dtype=torch.int32)
    intrinsics = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_to_world = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5]],
        dtype=torch.float32,
    )

    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).expand(images.shape[0], images.shape[1], -1, -1)
    camera_to_world = camera_to_world.unsqueeze(0).unsqueeze(0).expand(images.shape[0], images.shape[1], -1, -1)

    try:
        encoder = CLIP(model_name=model_name, tune_mode="freeze").to(device)
        encoder.eval()

        rgbd_batch = image_processor.process_rgbd(
            images=images,
            depths=depths,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )

        rgb = rgbd_batch.image.to(device)
        depth = rgbd_batch.depth.to(device)
        intrinsics = rgbd_batch.intrinsics.to(device)
        camera_to_world = None if rgbd_batch.camera_to_world is None else rgbd_batch.camera_to_world.to(device)

        with torch.no_grad():
            vision_out = encoder(rgb)
            geometry_out = encoder.backproject(
                depth=depth,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                depth_scale=1000.0,
                min_depth=0.01,
                max_depth=3.0,
            )
            feature_map = encoder.patch_tokens_to_featmap(
                vision_out["patch_tokens"],
                image_hw=rgb.shape[-2:],
            )

        print("rgb             :", tuple(rgb.shape))
        print("patch_tokens    :", tuple(vision_out["patch_tokens"].shape))
        print("global_token    :", tuple(vision_out["global_token"].shape))
        print("feature_map     :", tuple(feature_map.shape))
        print("patch_coords    :", tuple(geometry_out["patch_coords"].shape))
        print("patch_valid_mask:", tuple(geometry_out["patch_valid_mask"].shape))

    except Exception as error:
        print("clip example needs model weights and a valid transformers runtime.")
        print(error)


if __name__ == "__main__":
    example()