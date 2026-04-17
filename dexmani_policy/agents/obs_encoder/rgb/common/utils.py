import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union

HW = Tuple[int, int]
ArrayLike = Union[np.ndarray, torch.Tensor]


def to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError(f"Unsupported input type: {type(x)}")


def to_hw(size: Union[int, Tuple[int, int], None]) -> Optional[HW]:
    if size is None:
        return None
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("size should be an int or a tuple (H, W).")
    return int(size[0]), int(size[1])


def get_interpolation(name: str) -> str:
    name = name.lower()
    if name in {"bilinear", "bicubic", "nearest"}:
        return name
    raise ValueError(f"Unsupported interpolation: {name}")


def to_rgb_tensor(images: ArrayLike) -> torch.Tensor:
    images = to_tensor(images)
    if images.ndim < 3:
        raise ValueError("images should have at least 3 dims.")

    if images.shape[-3] != 3:
        if images.shape[-1] != 3:
            raise ValueError("Cannot infer RGB channel dimension.")
        images = images.movedim(-1, -3)

    if images.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        images = images.to(torch.float32).div_(255.0)
    else:
        images = images.to(torch.float32)
        image_min = float(images.amin().item())
        image_max = float(images.amax().item())
        if image_min < -1e-6 or image_max > 1.0 + 1e-6:
            raise ValueError(
                f"Float RGB images are expected to be in [0, 1], got value range [{image_min}, {image_max}]."
            )

    return images.contiguous()


def to_depth_tensor(depths: ArrayLike) -> torch.Tensor:
    depths = to_tensor(depths)
    if depths.ndim < 2:
        raise ValueError("depths should have at least 2 dims.")

    if depths.ndim >= 3 and depths.shape[-1] == 1:
        depths = depths.squeeze(-1)
    if depths.ndim < 3 or depths.shape[-3] != 1:
        depths = depths.unsqueeze(depths.ndim - 2)
    return depths.to(torch.float32).contiguous()


def flatten_batch(x: torch.Tensor, trailing_ndim: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if x.ndim < trailing_ndim:
        raise ValueError(f"Expected tensor with at least {trailing_ndim} dims, got {tuple(x.shape)}")
    leading_shape = tuple(x.shape[:-trailing_ndim])
    return x.reshape(-1, *x.shape[-trailing_ndim:]), leading_shape


def restore_batch(x: torch.Tensor, leading_shape: Tuple[int, ...]) -> torch.Tensor:
    if len(leading_shape) == 0:
        if x.shape[0] != 1:
            raise ValueError(f"Expected flattened batch size 1 for unbatched restore, got {tuple(x.shape)}")
        return x.reshape(*x.shape[1:])
    return x.reshape(*leading_shape, *x.shape[1:])


def make_spatial_transform_meta(
    orig_hw: HW,
    resized_hw: HW,
    processed_hw: HW,
    resize_scale_xy: Tuple[float, float],
    crop_top_left: Tuple[int, int],
) -> Dict[str, object]:
    return {
        "orig_hw": orig_hw,
        "resized_hw": resized_hw,
        "processed_hw": processed_hw,
        "resize_scale_xy": resize_scale_xy,
        "crop_top_left": crop_top_left,
    }


def broadcast_matrix(
    matrix: ArrayLike,
    mat_shape: Tuple[int, int],
    leading_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    matrix = to_tensor(matrix).to(device=device, dtype=dtype)

    if tuple(matrix.shape) == mat_shape:
        view_shape = (*([1] * len(leading_shape)), *mat_shape)
        if len(leading_shape) == 0:
            return matrix.reshape(*mat_shape)
        return matrix.reshape(*view_shape).expand(*leading_shape, *mat_shape).contiguous()

    if tuple(matrix.shape[-2:]) != mat_shape:
        raise ValueError(f"Expected matrix trailing shape {mat_shape}, got {tuple(matrix.shape[-2:])}.")

    matrix_leading_shape = tuple(matrix.shape[:-2])
    if len(matrix_leading_shape) > len(leading_shape):
        raise ValueError(
            f"Expected matrix shape {mat_shape} or leading dims compatible with {leading_shape}, "
            f"got {tuple(matrix.shape)}."
        )

    padded_leading_shape = matrix_leading_shape + (1,) * (len(leading_shape) - len(matrix_leading_shape))
    if not all(m == t or m == 1 for m, t in zip(padded_leading_shape, leading_shape)):
        raise ValueError(
            f"Expected matrix leading dims broadcastable to {leading_shape}, got {matrix_leading_shape}."
        )

    matrix = matrix.reshape(*padded_leading_shape, *mat_shape)
    if padded_leading_shape != leading_shape:
        matrix = matrix.expand(*leading_shape, *mat_shape)
    return matrix.contiguous()


def flatten_matrix_batch(
    matrix: ArrayLike,
    mat_shape: Tuple[int, int],
    leading_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    matrix = broadcast_matrix(
        matrix=matrix,
        mat_shape=mat_shape,
        leading_shape=leading_shape,
        device=device,
        dtype=dtype,
    )
    if len(leading_shape) == 0:
        return matrix.reshape(1, *mat_shape)
    return matrix.reshape(-1, *mat_shape)


def restore_matrix_batch(matrix_batch: torch.Tensor, leading_shape: Tuple[int, ...]) -> torch.Tensor:
    if matrix_batch.ndim == 2:
        return matrix_batch
    if len(leading_shape) == 0:
        if matrix_batch.shape[0] != 1:
            raise ValueError(f"Expected flattened batch size 1 for unbatched restore, got {tuple(matrix_batch.shape)}")
        return matrix_batch.reshape(*matrix_batch.shape[-2:])
    return matrix_batch.reshape(*leading_shape, *matrix_batch.shape[-2:])


def update_intrinsics_after_spatial_ops(
    intrinsics: torch.Tensor,
    scale_x: float,
    scale_y: float,
    crop_left: int,
    crop_top: int,
) -> torch.Tensor:
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 0] *= scale_x
    intrinsics[:, 1, 1] *= scale_y
    intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * scale_x - float(crop_left)
    intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * scale_y - float(crop_top)
    return intrinsics


def get_patch_grid_size(image_hw: HW, patch_size: int) -> HW:
    image_h, image_w = image_hw
    if patch_size <= 0:
        raise ValueError(f"patch_size should be positive, got {patch_size}.")
    if image_h % patch_size != 0 or image_w % patch_size != 0:
        raise ValueError(f"Input size {(image_h, image_w)} must be divisible by patch size {patch_size}.")
    return image_h // patch_size, image_w // patch_size


def resolve_patch_grid_size(
    image_hw: HW,
    patch_size: Optional[int] = None,
    patch_grid_size: Optional[HW] = None,
) -> HW:
    if (patch_size is None) == (patch_grid_size is None):
        raise ValueError("Specify exactly one of patch_size or patch_grid_size.")
    if patch_grid_size is not None:
        grid = to_hw(patch_grid_size)
        assert grid is not None
        if grid[0] <= 0 or grid[1] <= 0:
            raise ValueError(f"patch_grid_size should be positive, got {grid}.")
        return grid
    return get_patch_grid_size(image_hw=image_hw, patch_size=int(patch_size))


def reshape_patch_tokens_to_map(patch_tokens: torch.Tensor, patch_grid_size: HW) -> torch.Tensor:
    batch_size, token_num, channel_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    if token_num != grid_h * grid_w:
        raise ValueError(f"Patch token number {token_num} does not match patch grid size {patch_grid_size}.")
    return patch_tokens.reshape(batch_size, grid_h, grid_w, channel_dim).permute(0, 3, 1, 2).contiguous()