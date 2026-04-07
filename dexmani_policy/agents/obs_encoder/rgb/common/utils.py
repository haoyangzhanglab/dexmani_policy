import torch
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Union

HW = Tuple[int, int]
ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class MatrixBatchSpec:
    """Bookkeeping structure for batched camera matrices."""
    base: torch.Tensor
    has_leading: bool
    num_items: int
    leading_shape: Tuple[int, ...]
    storage_leading_shape: Tuple[int, ...]


@dataclass(frozen=True)
class SpatialTransformMeta:
    """Metadata produced by image-space resize / crop operations."""
    orig_hw: HW
    resized_hw: HW
    processed_hw: HW
    resize_scale_xy: Tuple[float, float]
    crop_top_left: Tuple[int, int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


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
    if name == "bilinear":
        return "bilinear"
    if name == "bicubic":
        return "bicubic"
    if name == "nearest":
        return "nearest"
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
        return x
    return x.reshape(*leading_shape, *x.shape[1:])


def prepare_matrix_batch(
    matrix: ArrayLike,
    mat_shape: Tuple[int, int],
    leading_shape: Tuple[int, ...],
    collapse_repeated: bool = True,
) -> MatrixBatchSpec:
    matrix = to_tensor(matrix).to(torch.float32)
    expected_shape = (*leading_shape, *mat_shape)
    num_items = int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1

    if tuple(matrix.shape) == mat_shape:
        return MatrixBatchSpec(
            base=matrix.view(1, *mat_shape).contiguous(),
            has_leading=False,
            num_items=num_items,
            leading_shape=leading_shape,
            storage_leading_shape=tuple(),
        )

    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Expected matrix shape {mat_shape} or {expected_shape}, got {tuple(matrix.shape)}.")

    storage_leading_shape = leading_shape
    base = matrix.reshape(-1, *mat_shape).contiguous()
    if collapse_repeated and len(leading_shape) > 0:
        grouped_matrix = matrix.reshape(*leading_shape, *mat_shape).contiguous()
        for prefix_len in range(len(leading_shape) + 1):
            prefix_shape = leading_shape[:prefix_len]
            suffix_num_items = int(np.prod(leading_shape[prefix_len:])) if prefix_len < len(leading_shape) else 1
            candidate = grouped_matrix.reshape(*prefix_shape, suffix_num_items, *mat_shape)
            repeated_reference = candidate[..., :1, :, :]
            if torch.equal(candidate, repeated_reference.expand_as(candidate)):
                storage_leading_shape = prefix_shape
                base = candidate[..., 0, :, :].reshape(-1, *mat_shape).contiguous()
                break

    return MatrixBatchSpec(
        base=base,
        has_leading=True,
        num_items=num_items,
        leading_shape=leading_shape,
        storage_leading_shape=storage_leading_shape,
    )


def _expand_collapsed_matrix_batch(matrix_batch: torch.Tensor, spec: MatrixBatchSpec) -> torch.Tensor:
    if spec.storage_leading_shape == spec.leading_shape:
        return matrix_batch

    mat_shape = tuple(matrix_batch.shape[-2:])
    matrix_batch = matrix_batch.reshape(*spec.storage_leading_shape, *mat_shape)
    expand_suffix_ndim = len(spec.leading_shape) - len(spec.storage_leading_shape)
    view_shape = (*spec.storage_leading_shape, *([1] * expand_suffix_ndim), *mat_shape)
    matrix_batch = matrix_batch.reshape(*view_shape)
    matrix_batch = matrix_batch.expand(*spec.leading_shape, *mat_shape)
    return matrix_batch.reshape(-1, *mat_shape)


def restore_matrix_batch(matrix_batch: torch.Tensor, spec: MatrixBatchSpec) -> torch.Tensor:
    if spec.has_leading:
        matrix_batch = _expand_collapsed_matrix_batch(matrix_batch, spec)
        return matrix_batch.reshape(*spec.leading_shape, *matrix_batch.shape[-2:])
    return matrix_batch[0]


def expand_matrix_batch(matrix_batch: torch.Tensor, spec: MatrixBatchSpec) -> torch.Tensor:
    return _expand_collapsed_matrix_batch(matrix_batch, spec)


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
        return to_hw(patch_grid_size)  # type: ignore[return-value]
    return get_patch_grid_size(image_hw=image_hw, patch_size=int(patch_size))


def reshape_patch_tokens_to_map(patch_tokens: torch.Tensor, patch_grid_size: HW) -> torch.Tensor:
    batch_size, token_num, channel_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    if token_num != grid_h * grid_w:
        raise ValueError(f"Patch token number {token_num} does not match patch grid size {patch_grid_size}.")
    return patch_tokens.view(batch_size, grid_h, grid_w, channel_dim).permute(0, 3, 1, 2).contiguous()
