import torch
import torch.nn as nn
import torchvision
from typing import Dict, Literal, Optional, Sequence

from dexmani_policy.agents.obs_encoder.rgb.common.image_processor import ImageProcessor
from dexmani_policy.agents.obs_encoder.rgb.common.geometry_processor import GeometryProcessor
from dexmani_policy.agents.obs_encoder.rgb.common.utils import (
    flatten_batch,
    restore_batch,
    reshape_patch_tokens_to_map,
)

TuneMode = Literal["freeze", "full"]
NormMode = Literal["frozen_bn", "group_norm"]
GlobalTokenType = Literal["avg"]


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "num_batches_tracked"
        if key in state_dict:
            del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        running_mean = self.running_mean.view(1, -1, 1, 1)
        running_var = self.running_var.view(1, -1, 1, 1)
        scale = weight * (running_var + 1e-5).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


def replace_batch_norm_with_group_norm(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_groups = min(32, child.num_features)
            while child.num_features % num_groups != 0:
                num_groups -= 1
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features))
        else:
            replace_batch_norm_with_group_norm(child)
    return module


class ResNet(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        tune_mode: TuneMode = "freeze",
        norm_mode: NormMode = "frozen_bn",
        global_token_type: GlobalTokenType = "avg",
        out_dim: Optional[int] = None,
        weights=None,
    ):
        super().__init__()

        self.model_name = model_name
        self.tune_mode = tune_mode
        self.global_token_type = global_token_type
        self.norm_mode = norm_mode
        self.output_stride = 32

        if not hasattr(torchvision.models, model_name):
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        norm_layer = FrozenBatchNorm2d if norm_mode == "frozen_bn" else nn.BatchNorm2d
        backbone = getattr(torchvision.models, model_name)(weights=weights, norm_layer=norm_layer)

        if norm_mode == "group_norm":
            backbone = replace_batch_norm_with_group_norm(backbone)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.hidden_dim = int(backbone.fc.in_features)
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        self.proj = nn.Identity() if self.out_dim == self.hidden_dim else nn.Conv2d(
            self.hidden_dim,
            self.out_dim,
            kernel_size=1,
        )
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

        raise ValueError(f"Unsupported tune_mode: {tune_mode}")

    def get_global_token(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.global_token_type == "avg":
            return feature_map.mean(dim=(-2, -1))

        raise ValueError(f"Unsupported global_token_type: {self.global_token_type}")

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.ndim < 4 or rgb.shape[-3] != 3:
            raise ValueError(f"rgb should have shape [..., 3, H, W], got {tuple(rgb.shape)}")

        if self.tune_mode == "freeze":
            self.backbone.eval()

        flat_rgb, leading_shape = flatten_batch(rgb, trailing_ndim=3)
        feature_map = self.backbone(flat_rgb)
        feature_map = self.proj(feature_map)

        patch_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        global_token = self.get_global_token(feature_map)

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
            collapse_repeated_camera=True,
        )

        patch_geometry = self.geometry_processor.pool_patch_coordinates(
            coords=dense_geometry["coords"],
            valid_mask=dense_geometry["valid_mask"],
            patch_size=self.output_stride,
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
        feature_h = (int(image_hw[0]) + self.output_stride - 1) // self.output_stride
        feature_w = (int(image_hw[1]) + self.output_stride - 1) // self.output_stride

        flat_patch_tokens, leading_shape = flatten_batch(patch_tokens, trailing_ndim=2)
        feature_map = reshape_patch_tokens_to_map(flat_patch_tokens, (feature_h, feature_w))
        return restore_batch(feature_map, leading_shape)

def example() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "resnet18"

    image_processor = ImageProcessor.from_preset("resnet")

    images = torch.randint(0, 256, (16, 2, 480, 640, 3), dtype=torch.uint8)
    depths = torch.randint(1, 2000, (16, 2, 480, 640), dtype=torch.uint16)
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
        encoder = ResNet(
            model_name=model_name,
            tune_mode="freeze",
            norm_mode="frozen_bn",
            out_dim=512,
            weights=None,
        ).to(device)
        encoder.eval()

        rgbd_batch = image_processor.process_rgbd(
            images=images,
            depths=depths,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )

        rgb = rgbd_batch["image"].to(device)
        depth = rgbd_batch["depth"].to(device)
        intrinsics = rgbd_batch["intrinsics"].to(device)
        camera_to_world = None if rgbd_batch["camera_to_world"] is None else rgbd_batch["camera_to_world"].to(device)

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
        print("resnet example failed.")
        print(error)


if __name__ == "__main__":
    example()