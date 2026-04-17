import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from dexmani_policy.agents.obs_encoder.rgb.common.utils import (
    ArrayLike,
    flatten_batch,
    flatten_matrix_batch,
    restore_batch,
    resolve_patch_grid_size,
    to_depth_tensor,
)


class GeometryProcessor:
    """Depth back-projection and patch-level geometry pooling."""
    def __init__(self):
        self.pixel_grid_cache: Dict[Tuple[int, int, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def clear_cache(self) -> None:
        self.pixel_grid_cache.clear()

    def get_pixel_grid(
        self,
        image_h: int,
        image_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (image_h, image_w, str(device), str(dtype))
        if key not in self.pixel_grid_cache:
            v, u = torch.meshgrid(
                torch.arange(image_h, device=device, dtype=dtype),
                torch.arange(image_w, device=device, dtype=dtype),
                indexing="ij",
            )
            self.pixel_grid_cache[key] = (u.unsqueeze(0), v.unsqueeze(0))
        return self.pixel_grid_cache[key]

    def backproject_depth(
        self,
        depth: ArrayLike,
        intrinsics: ArrayLike,
        camera_to_world: Optional[ArrayLike] = None,
        depth_scale: float = 1000.0,
        min_depth: float = 0.0,
        max_depth: Optional[float] = None,
        collapse_repeated_camera: bool = True,
    ) -> Dict[str, object]:
        _ = collapse_repeated_camera

        depth = to_depth_tensor(depth)
        flat_depth, leading_shape = flatten_batch(depth, trailing_ndim=3)
        batch_size, _, image_h, image_w = flat_depth.shape

        flat_intrinsics = flatten_matrix_batch(
            matrix=intrinsics,
            mat_shape=(3, 3),
            leading_shape=leading_shape,
            device=flat_depth.device,
        )

        flat_camera_to_world = None
        if camera_to_world is not None:
            flat_camera_to_world = flatten_matrix_batch(
                matrix=camera_to_world,
                mat_shape=(3, 4),
                leading_shape=leading_shape,
                device=flat_depth.device,
            )

        depth_metric = flat_depth / float(depth_scale)

        valid_mask = depth_metric > float(min_depth)
        if max_depth is not None:
            valid_mask = valid_mask & (depth_metric < float(max_depth))

        u, v = self.get_pixel_grid(image_h, image_w, device=flat_depth.device, dtype=flat_depth.dtype)

        fx = flat_intrinsics[:, 0, 0].reshape(batch_size, 1, 1)
        fy = flat_intrinsics[:, 1, 1].reshape(batch_size, 1, 1)
        cx = flat_intrinsics[:, 0, 2].reshape(batch_size, 1, 1)
        cy = flat_intrinsics[:, 1, 2].reshape(batch_size, 1, 1)

        eps = 1e-12
        if torch.any(fx.abs() < eps) or torch.any(fy.abs() < eps):
            raise ValueError("Invalid intrinsics: fx/fy must be non-zero.")

        z = depth_metric[:, 0]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        valid_mask_float = valid_mask.to(flat_depth.dtype)
        camera_coords = torch.stack([x, y, z], dim=1) * valid_mask_float

        coord_frame = "camera"
        coords = camera_coords
        if flat_camera_to_world is not None:
            rotation = flat_camera_to_world[:, :, :3]
            translation = flat_camera_to_world[:, :, 3:].contiguous()
            coords = torch.bmm(rotation, camera_coords.reshape(batch_size, 3, -1)) + translation
            coords = coords.reshape(batch_size, 3, image_h, image_w) * valid_mask_float
            coord_frame = "world"

        return {
            "coords": restore_batch(coords, leading_shape),
            "valid_mask": restore_batch(valid_mask, leading_shape),
            "coord_frame": coord_frame,
            "depth_scale": float(depth_scale),
            "min_depth": float(min_depth),
            "max_depth": None if max_depth is None else float(max_depth),
        }

    def pool_patch_coordinates(
        self,
        coords: torch.Tensor,
        valid_mask: torch.Tensor,
        patch_size: Optional[int] = None,
        patch_grid_size: Optional[Tuple[int, int]] = None,
        min_valid_ratio: float = 0.25,
    ) -> Dict[str, object]:
        if coords.ndim < 4 or coords.shape[-3] != 3:
            raise ValueError(f"coords should have shape [..., 3, H, W], got {tuple(coords.shape)}")
        if valid_mask.ndim < 4 or valid_mask.shape[-3] != 1:
            raise ValueError(f"valid_mask should have shape [..., 1, H, W], got {tuple(valid_mask.shape)}")
        if coords.shape[:-3] != valid_mask.shape[:-3] or coords.shape[-2:] != valid_mask.shape[-2:]:
            raise ValueError("coords and valid_mask should share the same leading dims and spatial size.")
        if not (0.0 <= min_valid_ratio <= 1.0):
            raise ValueError(f"min_valid_ratio should be in [0, 1], got {min_valid_ratio}")

        image_h, image_w = coords.shape[-2:]
        grid_h, grid_w = resolve_patch_grid_size(
            image_hw=(image_h, image_w),
            patch_size=patch_size,
            patch_grid_size=patch_grid_size,
        )
        if image_h % grid_h != 0 or image_w % grid_w != 0:
            raise ValueError(f"Image size {(image_h, image_w)} is not divisible by patch grid size {(grid_h, grid_w)}.")

        flat_coords, leading_shape = flatten_batch(coords, trailing_ndim=3)
        flat_valid_mask, mask_leading_shape = flatten_batch(valid_mask, trailing_ndim=3)
        if leading_shape != mask_leading_shape:
            raise ValueError("coords and valid_mask should share the same leading dims.")

        kernel_h = image_h // grid_h
        kernel_w = image_w // grid_w
        flat_coords = flat_coords.float()
        flat_valid_mask = flat_valid_mask.float()

        coord_num = F.avg_pool2d(
            flat_coords * flat_valid_mask,
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
        )
        coord_den = F.avg_pool2d(
            flat_valid_mask,
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
        )

        coord_map = coord_num / coord_den.clamp_min(1e-6)
        coord_valid_map = coord_den >= float(min_valid_ratio)

        patch_coords = coord_map.flatten(2).transpose(1, 2).contiguous()
        patch_valid_mask = coord_valid_map.flatten(2).transpose(1, 2).contiguous()

        return {
            "patch_coords": restore_batch(patch_coords, leading_shape),
            "patch_valid_mask": restore_batch(patch_valid_mask, leading_shape),
            "patch_grid_size": (grid_h, grid_w),
            "patch_hw": (kernel_h, kernel_w),
        }


def example() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    geometry = GeometryProcessor()

    depth = torch.randint(1, 2000, (2, 4, 224, 224), dtype=torch.int32, device=device)
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

    dense = geometry.backproject_depth(
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        depth_scale=1000.0,
        min_depth=0.01,
        max_depth=3.0,
    )
    pooled = geometry.pool_patch_coordinates(
        coords=dense["coords"],
        valid_mask=dense["valid_mask"],
        patch_grid_size=(16, 16),
    )

    print("coords           :", tuple(dense["coords"].shape), dense["coord_frame"])
    print("valid_mask       :", tuple(dense["valid_mask"].shape))
    print("patch_coords     :", tuple(pooled["patch_coords"].shape))
    print("patch_valid_mask :", tuple(pooled["patch_valid_mask"].shape))


if __name__ == "__main__":
    example()