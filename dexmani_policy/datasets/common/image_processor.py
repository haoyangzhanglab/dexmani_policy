import numpy as np
import torch
from typing import Dict, Optional, Sequence, Tuple, Union
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


def to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError(f"Unsupported input type: {type(x)}")


def to_rgb_tensor(images: ArrayLike) -> torch.Tensor:
    images = to_tensor(images)
    if images.ndim < 3:
        raise ValueError("images should have at least 3 dims.")

    if images.shape[-3] == 3:
        pass
    elif images.shape[-1] == 3:
        images = images.movedim(-1, -3)
    else:
        raise ValueError("Cannot infer RGB channel dimension.")

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
) -> Dict[str, Union[torch.Tensor, bool, int, Tuple[int, ...]]]:
    matrix = to_tensor(matrix).to(torch.float32)
    expected_shape = (*leading_shape, *mat_shape)

    if tuple(matrix.shape) == mat_shape:
        base = matrix.view(1, *mat_shape).contiguous()
        return {
            "base": base,
            "has_leading": False,
            "num_items": int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1,
            "leading_shape": leading_shape,
        }

    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Expected matrix shape {mat_shape} or {expected_shape}, got {tuple(matrix.shape)}.")

    base = matrix.reshape(-1, *mat_shape).contiguous()
    if collapse_repeated and base.shape[0] > 1:
        if torch.equal(base, base[:1].expand_as(base)):
            base = base[:1].contiguous()

    return {
        "base": base,
        "has_leading": True,
        "num_items": int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1,
        "leading_shape": leading_shape,
    }


def restore_matrix_batch(
    matrix_batch: torch.Tensor,
    spec: Dict[str, Union[torch.Tensor, bool, int, Tuple[int, ...]]],
) -> torch.Tensor:
    has_leading = bool(spec["has_leading"])
    num_items = int(spec["num_items"])
    leading_shape = tuple(spec["leading_shape"])

    if has_leading:
        if matrix_batch.shape[0] == 1 and num_items > 1:
            matrix_batch = matrix_batch.expand(num_items, *matrix_batch.shape[1:])
        return matrix_batch.reshape(*leading_shape, *matrix_batch.shape[-2:])

    return matrix_batch[0]


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


class ImageProcessor:
    """统一的图像空间对齐模块。"""

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
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32)
        self.image_std = torch.tensor(image_std, dtype=torch.float32)
        self.interpolation = get_interpolation(interpolation)

        if self.image_size is not None and self.resize_shortest_edge is not None:
            raise ValueError("Use either image_size or resize_shortest_edge, not both.")

    def apply_spatial_ops(
        self,
        image_batch: torch.Tensor,
        depth_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Union[Tuple[int, int], Tuple[float, float]]]]:
        if image_batch.ndim != 4 or image_batch.shape[1] != 3:
            raise ValueError("image_batch should have shape [N, 3, H, W].")
        if depth_batch is not None and (depth_batch.ndim != 4 or depth_batch.shape[1] != 1):
            raise ValueError("depth_batch should have shape [N, 1, H, W].")

        orig_h, orig_w = image_batch.shape[-2:]
        resized_h, resized_w = orig_h, orig_w
        scale_x, scale_y = 1.0, 1.0

        if self.resize_shortest_edge is not None:
            scale = float(self.resize_shortest_edge) / float(min(orig_h, orig_w))
            resized_h = int(round(orig_h * scale))
            resized_w = int(round(orig_w * scale))
        elif self.image_size is not None:
            resized_h, resized_w = self.image_size

        if resized_h != orig_h or resized_w != orig_w:
            scale_x = float(resized_w) / float(orig_w)
            scale_y = float(resized_h) / float(orig_h)
            image_batch = TF.resize(
                image_batch,
                [resized_h, resized_w],
                interpolation=self.interpolation,
                antialias=True,
            )
            if depth_batch is not None:
                depth_batch = TF.resize(
                    depth_batch,
                    [resized_h, resized_w],
                    interpolation=InterpolationMode.NEAREST,
                )

        crop_top, crop_left = 0, 0
        processed_h, processed_w = resized_h, resized_w

        if self.center_crop_size is not None:
            crop_h, crop_w = self.center_crop_size
            if crop_h > resized_h or crop_w > resized_w:
                raise ValueError(
                    f"Crop size {(crop_h, crop_w)} is larger than image size {(resized_h, resized_w)}."
                )
            crop_top = int(round((resized_h - crop_h) / 2.0))
            crop_left = int(round((resized_w - crop_w) / 2.0))
            processed_h, processed_w = crop_h, crop_w
            image_batch = image_batch[..., crop_top:crop_top + crop_h, crop_left:crop_left + crop_w].contiguous()
            if depth_batch is not None:
                depth_batch = depth_batch[..., crop_top:crop_top + crop_h, crop_left:crop_left + crop_w].contiguous()

        meta = {
            "orig_hw": (orig_h, orig_w),
            "resized_hw": (resized_h, resized_w),
            "processed_hw": (processed_h, processed_w),
            "resize_scale_xy": (scale_x, scale_y),
            "crop_top_left": (crop_top, crop_left),
        }
        return image_batch, depth_batch, meta

    def normalize_image(self, image_batch: torch.Tensor) -> torch.Tensor:
        mean = self.image_mean.to(device=image_batch.device, dtype=image_batch.dtype).view(1, 3, 1, 1)
        std = self.image_std.to(device=image_batch.device, dtype=image_batch.dtype).view(1, 3, 1, 1)
        return image_batch.sub(mean).div(std)

    def process_image(self, images: ArrayLike) -> Dict[str, Union[torch.Tensor, Dict]]:
        images = to_rgb_tensor(images)
        image_batch, leading_shape = flatten_batch(images, trailing_ndim=3)
        image_batch, _, meta = self.apply_spatial_ops(image_batch)
        image_batch = self.normalize_image(image_batch)
        meta["leading_shape"] = leading_shape
        return {
            "image": restore_batch(image_batch, leading_shape),
            "meta": meta,
        }

    def process_rgbd(
        self,
        images: ArrayLike,
        depths: ArrayLike,
        intrinsics: ArrayLike,
        camera_to_world: Optional[ArrayLike] = None,
        collapse_repeated_camera: bool = True,
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        images = to_rgb_tensor(images)
        depths = to_depth_tensor(depths)

        image_batch, leading_shape = flatten_batch(images, trailing_ndim=3)
        depth_batch, depth_leading_shape = flatten_batch(depths, trailing_ndim=3)
        if leading_shape != depth_leading_shape:
            raise ValueError(f"images leading shape {leading_shape} != depths leading shape {depth_leading_shape}")

        intr_spec = prepare_matrix_batch(
            intrinsics,
            mat_shape=(3, 3),
            leading_shape=leading_shape,
            collapse_repeated=collapse_repeated_camera,
        )

        c2w_spec = None
        if camera_to_world is not None:
            c2w_spec = prepare_matrix_batch(
                camera_to_world,
                mat_shape=(3, 4),
                leading_shape=leading_shape,
                collapse_repeated=collapse_repeated_camera,
            )

        image_batch, depth_batch, meta = self.apply_spatial_ops(image_batch, depth_batch)
        image_batch = self.normalize_image(image_batch)

        scale_x, scale_y = meta["resize_scale_xy"]
        crop_top, crop_left = meta["crop_top_left"]
        intrinsics_flat = update_intrinsics_after_spatial_ops(
            intrinsics=intr_spec["base"],
            scale_x=scale_x,
            scale_y=scale_y,
            crop_left=crop_left,
            crop_top=crop_top,
        )

        out: Dict[str, Union[torch.Tensor, Dict]] = {
            "image": restore_batch(image_batch, leading_shape),
            "depth": restore_batch(depth_batch, leading_shape),
            "intrinsics": restore_matrix_batch(intrinsics_flat, intr_spec),
            "meta": {
                **meta,
                "leading_shape": leading_shape,
                "camera_shared_across_time": intr_spec["base"].shape[0] == 1,
            },
        }

        if c2w_spec is not None:
            out["camera_to_world"] = restore_matrix_batch(c2w_spec["base"], c2w_spec)
            out["meta"]["camera_to_world_shared_across_time"] = c2w_spec["base"].shape[0] == 1

        return out


def example():
    processor = build_image_processor("dino")

    t, h, w = 4, 480, 640
    images = torch.randint(0, 256, (t, h, w, 3), dtype=torch.uint8)
    depths = torch.randint(1, 2000, (t, h, w), dtype=torch.int32)

    intrinsics = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_to_world = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5]],
        dtype=torch.float32,
    )

    rgb_out = processor.process_image(images)
    print("process_image image:", tuple(rgb_out["image"].shape))

    rgbd_out = processor.process_rgbd(images, depths, intrinsics, camera_to_world)
    print("process_rgbd image:", tuple(rgbd_out["image"].shape))
    print("process_rgbd depth:", tuple(rgbd_out["depth"].shape))
    print("process_rgbd intrinsics:", tuple(rgbd_out["intrinsics"].shape))
    print("process_rgbd camera_to_world:", tuple(rgbd_out["camera_to_world"].shape))


if __name__ == "__main__":
    example()