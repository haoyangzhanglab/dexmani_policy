from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError(f"Unsupported input type: {type(x)}")


def to_depth_tensor(depths: ArrayLike) -> torch.Tensor:
    depths = to_tensor(depths)
    if depths.ndim < 2:
        raise ValueError("depths should have at least 2 dims.")

    if depths.ndim >= 3 and depths.shape[-1] == 1:
        depths = depths.squeeze(-1)
    if depths.ndim >= 3 and depths.shape[-3] == 1:
        pass
    else:
        depths = depths.unsqueeze(depths.ndim - 2)

    return depths.to(torch.float32).contiguous()


def flatten_batch(x: torch.Tensor, trailing_ndim: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    leading_shape = tuple(x.shape[:-trailing_ndim])
    x = x.reshape(-1, *x.shape[-trailing_ndim:])
    return x, leading_shape


def restore_batch(x: torch.Tensor, leading_shape: Tuple[int, ...]) -> torch.Tensor:
    if len(leading_shape) == 0:
        return x.reshape(*x.shape[1:])
    return x.reshape(*leading_shape, *x.shape[1:])


def prepare_matrix_batch(
    matrix: ArrayLike,
    mat_shape: Tuple[int, int],
    leading_shape: Tuple[int, ...],
    collapse_repeated: bool = True,
) -> torch.Tensor:
    matrix = to_tensor(matrix).to(torch.float32)
    expected_shape = (*leading_shape, *mat_shape)

    if tuple(matrix.shape) == mat_shape:
        return matrix.view(1, *mat_shape).contiguous()
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Expected matrix shape {mat_shape} or {expected_shape}, got {tuple(matrix.shape)}.")

    matrix = matrix.reshape(-1, *mat_shape).contiguous()
    if collapse_repeated and matrix.shape[0] > 1:
        if torch.equal(matrix, matrix[:1].expand_as(matrix)):
            matrix = matrix[:1].contiguous()
    return matrix


class WorldCoordGenerator:
    """独立几何模块: depth + intrinsics + camera_to_world -> dense world coords。"""

    def __init__(self):
        self.grid_cache = {}

    def get_pixel_grid(
        self,
        image_h: int,
        image_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (image_h, image_w, str(device), str(dtype))
        if key not in self.grid_cache:
            v, u = torch.meshgrid(
                torch.arange(image_h, device=device, dtype=dtype),
                torch.arange(image_w, device=device, dtype=dtype),
                indexing="ij",
            )
            self.grid_cache[key] = (u.unsqueeze(0), v.unsqueeze(0))
        return self.grid_cache[key]

    def compute(
        self,
        depths: ArrayLike,
        intrinsics: ArrayLike,
        camera_to_world: Optional[ArrayLike] = None,
        depth_scale: float = 1000.0,
        min_depth: float = 0.0,
        max_depth: Optional[float] = None,
        collapse_repeated_camera: bool = True,
    ) -> Dict[str, torch.Tensor]:
        depths = to_depth_tensor(depths)
        depth_batch, leading_shape = flatten_batch(depths, trailing_ndim=3)

        intrinsics_batch = prepare_matrix_batch(
            intrinsics,
            mat_shape=(3, 3),
            leading_shape=leading_shape,
            collapse_repeated=collapse_repeated_camera,
        )
        if intrinsics_batch.shape[0] == 1 and depth_batch.shape[0] > 1:
            intrinsics_batch = intrinsics_batch.expand(depth_batch.shape[0], -1, -1)

        camera_to_world_batch = None
        if camera_to_world is not None:
            camera_to_world_batch = prepare_matrix_batch(
                camera_to_world,
                mat_shape=(3, 4),
                leading_shape=leading_shape,
                collapse_repeated=collapse_repeated_camera,
            )
            if camera_to_world_batch.shape[0] == 1 and depth_batch.shape[0] > 1:
                camera_to_world_batch = camera_to_world_batch.expand(depth_batch.shape[0], -1, -1)

        n, _, image_h, image_w = depth_batch.shape
        device = depth_batch.device
        dtype = depth_batch.dtype

        depth_metric = depth_batch / float(depth_scale)
        valid_mask = depth_metric > float(min_depth)
        if max_depth is not None:
            valid_mask = valid_mask & (depth_metric < float(max_depth))
        valid_mask = valid_mask.to(dtype)

        u, v = self.get_pixel_grid(image_h, image_w, device=device, dtype=dtype)

        fx = intrinsics_batch[:, 0, 0].view(n, 1, 1)
        fy = intrinsics_batch[:, 1, 1].view(n, 1, 1)
        cx = intrinsics_batch[:, 0, 2].view(n, 1, 1)
        cy = intrinsics_batch[:, 1, 2].view(n, 1, 1)

        z = depth_metric[:, 0]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        camera_coords = torch.stack([x, y, z], dim=1) * valid_mask

        if camera_to_world_batch is None:
            world_coords = camera_coords
        else:
            rotation = camera_to_world_batch[:, :, :3]
            translation = camera_to_world_batch[:, :, 3:].contiguous()
            flat_camera = camera_coords.reshape(n, 3, -1)
            flat_world = torch.bmm(rotation, flat_camera) + translation
            world_coords = flat_world.reshape(n, 3, image_h, image_w) * valid_mask

        return {
            "image_world_coords": restore_batch(world_coords, leading_shape),
            "image_valid_mask": restore_batch(valid_mask, leading_shape),
        }


def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    geometry = WorldCoordGenerator()

    depths = torch.randint(1, 2000, (4, 224, 224), dtype=torch.int32, device=device)
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

    out = geometry.compute(depths, intrinsics, camera_to_world, depth_scale=1000.0, min_depth=0.01, max_depth=3.0)
    print("image_world_coords:", tuple(out["image_world_coords"].shape))
    print("image_valid_mask:", tuple(out["image_valid_mask"].shape))


if __name__ == "__main__":
    example()