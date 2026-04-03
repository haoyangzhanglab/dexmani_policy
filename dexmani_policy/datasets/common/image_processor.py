from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

ArrayLike = Union[np.ndarray, torch.Tensor]


PROCESS_CONFIGS: Dict[str, Dict] = {
    "dino": {
        "resize_shortest_edge": 256,
        "center_crop_size": 224,
        "image_mean": (0.485, 0.456, 0.406),
        "image_std": (0.229, 0.224, 0.225),
        "interpolation": "bicubic",
    },
    "resnet": {
        "resize_shortest_edge": 256,
        "center_crop_size": 224,
        "image_mean": (0.485, 0.456, 0.406),
        "image_std": (0.229, 0.224, 0.225),
        "interpolation": "bilinear",
    },
    "clip": {
        "resize_shortest_edge": 224,
        "center_crop_size": 224,
        "image_mean": (0.48145466, 0.45782750, 0.40821073),
        "image_std": (0.26862954, 0.26130258, 0.27577711),
        "interpolation": "bicubic",
    },
    "siglip": {
        "image_size": (224, 224),
        "center_crop_size": None,
        "image_mean": (0.5, 0.5, 0.5),
        "image_std": (0.5, 0.5, 0.5),
        "interpolation": "bicubic",
    },
}


REALSENSE_RGB_SIZES = {
    "D435": ((480, 640), (720, 1280), (1080, 1920)),
    "L515": ((480, 640), (720, 1280), (1080, 1920)),
}


def to_hw(size: Union[int, Tuple[int, int], None]) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("size should be an int or a tuple (H, W).")
    return int(size[0]), int(size[1])


def get_interpolation(name: str) -> InterpolationMode:
    name = name.lower()
    if name == "bilinear":
        return InterpolationMode.BILINEAR
    if name == "bicubic":
        return InterpolationMode.BICUBIC
    if name == "nearest":
        return InterpolationMode.NEAREST
    raise ValueError(f"Unsupported interpolation: {name}")


def build_image_processor(name: str) -> "ImageProcessor":
    if name not in PROCESS_CONFIGS:
        raise KeyError(f"Unknown process config: {name}")
    config = PROCESS_CONFIGS[name]
    return ImageProcessor(
        image_size=config.get("image_size"),
        resize_shortest_edge=config.get("resize_shortest_edge"),
        center_crop_size=config.get("center_crop_size"),
        image_mean=config["image_mean"],
        image_std=config["image_std"],
        interpolation=config.get("interpolation", "bilinear"),
    )


def to_rgb_tensor(image: ArrayLike) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if image.ndim != 3:
        raise ValueError("image should have shape [H, W, 3] or [3, H, W].")

    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1).contiguous()
    elif image.shape[0] != 3:
        raise ValueError("image should have 3 channels.")

    image = image.to(torch.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def to_depth_tensor(depth: ArrayLike) -> torch.Tensor:
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth)

    if depth.ndim == 2:
        depth = depth.unsqueeze(0)

    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError("depth should have shape [H, W] or [1, H, W].")

    return depth.to(torch.float32)


def to_matrix(matrix: ArrayLike, shape: Tuple[int, int]) -> torch.Tensor:
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    matrix = matrix.to(torch.float32)
    if tuple(matrix.shape) != shape:
        raise ValueError(f"matrix should have shape {shape}, got {tuple(matrix.shape)}.")
    return matrix


def resize_intrinsics(
    intrinsics: torch.Tensor,
    old_hw: Tuple[int, int],
    new_hw: Tuple[int, int],
) -> torch.Tensor:
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    scale_x = float(new_w) / float(old_w)
    scale_y = float(new_h) / float(old_h)

    intrinsics = intrinsics.clone()
    intrinsics[0, 0] *= scale_x
    intrinsics[1, 1] *= scale_y
    intrinsics[0, 2] *= scale_x
    intrinsics[1, 2] *= scale_y
    return intrinsics


def resize_shortest_edge(
    image: torch.Tensor,
    depth: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    shortest_edge: int,
    interpolation: InterpolationMode,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    old_h, old_w = image.shape[-2:]
    scale = float(shortest_edge) / float(min(old_h, old_w))
    new_h = int(round(old_h * scale))
    new_w = int(round(old_w * scale))

    image = TF.resize(image, [new_h, new_w], interpolation=interpolation, antialias=True)
    if depth is not None:
        depth = TF.resize(depth, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
    intrinsics = resize_intrinsics(intrinsics, (old_h, old_w), (new_h, new_w))
    return image, depth, intrinsics


def resize_to_size(
    image: torch.Tensor,
    depth: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    interpolation: InterpolationMode,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    old_h, old_w = image.shape[-2:]
    new_h, new_w = image_size

    image = TF.resize(image, [new_h, new_w], interpolation=interpolation, antialias=True)
    if depth is not None:
        depth = TF.resize(depth, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
    intrinsics = resize_intrinsics(intrinsics, (old_h, old_w), (new_h, new_w))
    return image, depth, intrinsics


def center_crop(
    image: torch.Tensor,
    depth: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    crop_size: Tuple[int, int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    crop_h, crop_w = crop_size
    image_h, image_w = image.shape[-2:]

    if crop_h > image_h or crop_w > image_w:
        raise ValueError(
            f"Crop size {(crop_h, crop_w)} is larger than image size {(image_h, image_w)}."
        )

    top = int(round((image_h - crop_h) / 2.0))
    left = int(round((image_w - crop_w) / 2.0))

    image = TF.crop(image, top, left, crop_h, crop_w)
    if depth is not None:
        depth = TF.crop(depth, top, left, crop_h, crop_w)

    intrinsics = intrinsics.clone()
    intrinsics[0, 2] -= float(left)
    intrinsics[1, 2] -= float(top)
    return image, depth, intrinsics


def normalize_image(image: torch.Tensor, image_mean: torch.Tensor, image_std: torch.Tensor) -> torch.Tensor:
    mean = image_mean.to(device=image.device, dtype=image.dtype)
    std = image_std.to(device=image.device, dtype=image.dtype)
    return (image - mean) / std


def depth_to_world_coords(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor],
    depth_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError("depth should have shape [1, H, W].")

    depth_metric = depth.to(torch.float32) / float(depth_scale)
    valid_mask = (depth_metric > 0).to(torch.float32)

    _, image_h, image_w = depth_metric.shape
    device = depth_metric.device
    dtype = depth_metric.dtype

    v, u = torch.meshgrid(
        torch.arange(image_h, device=device, dtype=dtype),
        torch.arange(image_w, device=device, dtype=dtype),
        indexing="ij",
    )

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    z = depth_metric[0]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    camera_coords = torch.stack([x, y, z], dim=0)

    if extrinsics is None:
        return camera_coords, valid_mask

    flat_camera = camera_coords.reshape(3, -1)
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3:4]
    world = rotation @ flat_camera + translation
    return world.view_as(camera_coords), valid_mask



class ImageProcessor:
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int], None] = None,
        resize_shortest_edge: Optional[int] = None,
        center_crop_size: Union[int, Tuple[int, int], None] = None,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        interpolation: str = "bilinear",
    ):
        self.image_size = to_hw(image_size)
        self.resize_shortest_edge = resize_shortest_edge
        self.center_crop_size = to_hw(center_crop_size)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.interpolation = get_interpolation(interpolation)

        if self.image_size is not None and self.resize_shortest_edge is not None:
            raise ValueError("Use either image_size or resize_shortest_edge, not both.")

    def preprocess(
        self,
        image: ArrayLike,
        depth: Optional[ArrayLike] = None,
        intrinsics: Optional[ArrayLike] = None,
        extrinsics: Optional[ArrayLike] = None,
        depth_scale: float = 1000.0,
    ) -> Dict[str, torch.Tensor]:
        image = to_rgb_tensor(image)
        depth = None if depth is None else to_depth_tensor(depth)

        if intrinsics is None:
            raise ValueError("intrinsics must be provided.")
        intrinsics = to_matrix(intrinsics, (3, 3))
        extrinsics = None if extrinsics is None else to_matrix(extrinsics, (4, 4))

        if self.resize_shortest_edge is not None:
            image, depth, intrinsics = resize_shortest_edge(
                image=image,
                depth=depth,
                intrinsics=intrinsics,
                shortest_edge=self.resize_shortest_edge,
                interpolation=self.interpolation,
            )
        elif self.image_size is not None:
            image, depth, intrinsics = resize_to_size(
                image=image,
                depth=depth,
                intrinsics=intrinsics,
                image_size=self.image_size,
                interpolation=self.interpolation,
            )

        if self.center_crop_size is not None:
            image, depth, intrinsics = center_crop(
                image=image,
                depth=depth,
                intrinsics=intrinsics,
                crop_size=self.center_crop_size,
            )

        image = normalize_image(image, self.image_mean, self.image_std)

        output = {
            "image": image,
            "intrinsics": intrinsics,
        }

        if depth is None:
            return output

        image_world_coords, image_valid_mask = depth_to_world_coords(
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_scale=depth_scale,
        )

        output.update(
            {
                "depth_resized": depth,
                "image_world_coords": image_world_coords,
                "image_valid_mask": image_valid_mask,
            }
        )
        return output