import torch
import torch.nn.functional as F
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
from dexmani_policy.agents.obs_encoder.rgb.common.utils import *


IMAGE_PROCESSOR_PRESETS: Dict[str, Dict[str, object]] = {
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


@dataclass(frozen=True)
class ImageProcessMeta:
    spatial: SpatialTransformMeta
    leading_shape: Tuple[int, ...]

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["spatial"] = self.spatial.to_dict()
        return out


@dataclass(frozen=True)
class RGBDProcessMeta(ImageProcessMeta):
    camera_shared_across_items: bool
    camera_to_world_shared_across_items: Optional[bool] = None


@dataclass(frozen=True)
class ProcessedImageBatch:
    image: torch.Tensor
    meta: ImageProcessMeta

    def to_dict(self) -> Dict[str, object]:
        return {"image": self.image, "meta": self.meta.to_dict()}


@dataclass(frozen=True)
class ProcessedRGBDBatch:
    image: torch.Tensor
    depth: torch.Tensor
    intrinsics: torch.Tensor
    meta: RGBDProcessMeta
    camera_to_world: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "image": self.image,
            "depth": self.depth,
            "intrinsics": self.intrinsics,
            "meta": self.meta.to_dict(),
        }
        if self.camera_to_world is not None:
            out["camera_to_world"] = self.camera_to_world
        return out


class ImageProcessor:
    """Unified image-space preprocessing for RGB and RGB-D observations."""
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

    @classmethod
    def from_preset(cls, name: str) -> "ImageProcessor":
        if name not in IMAGE_PROCESSOR_PRESETS:
            raise KeyError(f"Unknown image processor preset: {name}")
        return cls(**IMAGE_PROCESSOR_PRESETS[name])

    def apply_spatial_transform(
        self,
        image_batch: torch.Tensor,
        depth_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], SpatialTransformMeta]:
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
            image_batch = self._resize_tensor(image_batch, size=(resized_h, resized_w), mode=self.interpolation)
            if depth_batch is not None:
                depth_batch = self._resize_tensor(depth_batch, size=(resized_h, resized_w), mode="nearest")

        crop_top, crop_left = 0, 0
        processed_h, processed_w = resized_h, resized_w

        if self.center_crop_size is not None:
            crop_h, crop_w = self.center_crop_size
            if crop_h > resized_h or crop_w > resized_w:
                raise ValueError(f"Crop size {(crop_h, crop_w)} is larger than image size {(resized_h, resized_w)}.")

            crop_top = int(round((resized_h - crop_h) / 2.0))
            crop_left = int(round((resized_w - crop_w) / 2.0))
            processed_h, processed_w = crop_h, crop_w

            image_batch = image_batch[..., crop_top:crop_top + crop_h, crop_left:crop_left + crop_w].contiguous()
            if depth_batch is not None:
                depth_batch = depth_batch[..., crop_top:crop_top + crop_h, crop_left:crop_left + crop_w].contiguous()

        meta = SpatialTransformMeta(
            orig_hw=(orig_h, orig_w),
            resized_hw=(resized_h, resized_w),
            processed_hw=(processed_h, processed_w),
            resize_scale_xy=(scale_x, scale_y),
            crop_top_left=(crop_top, crop_left),
        )
        return image_batch, depth_batch, meta

    @staticmethod
    def _resize_tensor(
        x: torch.Tensor,
        size: Tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        if mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError(f"Unsupported interpolation mode: {mode}")
        if mode == "nearest":
            return F.interpolate(x, size=size, mode=mode)
        return F.interpolate(x, size=size, mode=mode, align_corners=False, antialias=True)

    def normalize(self, image_batch: torch.Tensor) -> torch.Tensor:
        mean = self.image_mean.to(device=image_batch.device, dtype=image_batch.dtype).view(1, 3, 1, 1)
        std = self.image_std.to(device=image_batch.device, dtype=image_batch.dtype).view(1, 3, 1, 1)
        return image_batch.sub(mean).div(std)

    def process_images(self, images: ArrayLike) -> ProcessedImageBatch:
        images = to_rgb_tensor(images)
        flat_images, leading_shape = flatten_batch(images, trailing_ndim=3)

        flat_images, _, spatial_meta = self.apply_spatial_transform(flat_images)
        flat_images = self.normalize(flat_images)

        return ProcessedImageBatch(
            image=restore_batch(flat_images, leading_shape),
            meta=ImageProcessMeta(spatial=spatial_meta, leading_shape=leading_shape),
        )

    def process_rgbd(
        self,
        images: ArrayLike,
        depths: ArrayLike,
        intrinsics: ArrayLike,
        camera_to_world: Optional[ArrayLike] = None,
        collapse_repeated_camera: bool = True,
    ) -> ProcessedRGBDBatch:
        images = to_rgb_tensor(images)
        depths = to_depth_tensor(depths)

        flat_images, leading_shape = flatten_batch(images, trailing_ndim=3)
        flat_depths, depth_leading_shape = flatten_batch(depths, trailing_ndim=3)
        if leading_shape != depth_leading_shape:
            raise ValueError(f"images leading shape {leading_shape} != depths leading shape {depth_leading_shape}")

        intrinsics_spec = prepare_matrix_batch(
            intrinsics,
            mat_shape=(3, 3),
            leading_shape=leading_shape,
            collapse_repeated=collapse_repeated_camera,
        )

        camera_to_world_spec = None
        if camera_to_world is not None:
            camera_to_world_spec = prepare_matrix_batch(
                camera_to_world,
                mat_shape=(3, 4),
                leading_shape=leading_shape,
                collapse_repeated=collapse_repeated_camera,
            )

        flat_images, flat_depths, spatial_meta = self.apply_spatial_transform(flat_images, flat_depths)
        flat_images = self.normalize(flat_images)

        scale_x, scale_y = spatial_meta.resize_scale_xy
        crop_top, crop_left = spatial_meta.crop_top_left
        flat_intrinsics = update_intrinsics_after_spatial_ops(
            intrinsics=intrinsics_spec.base,
            scale_x=scale_x,
            scale_y=scale_y,
            crop_left=crop_left,
            crop_top=crop_top,
        )

        return ProcessedRGBDBatch(
            image=restore_batch(flat_images, leading_shape),
            depth=restore_batch(flat_depths, leading_shape),
            intrinsics=restore_matrix_batch(flat_intrinsics, intrinsics_spec),
            camera_to_world=(
                restore_matrix_batch(camera_to_world_spec.base, camera_to_world_spec)
                if camera_to_world_spec is not None
                else None
            ),
            meta=RGBDProcessMeta(
                spatial=spatial_meta,
                leading_shape=leading_shape,
                camera_shared_across_items=intrinsics_spec.base.shape[0] == 1,
                camera_to_world_shared_across_items=(
                    camera_to_world_spec.base.shape[0] == 1 if camera_to_world_spec is not None else None
                ),
            ),
        )


def build_image_processor(name: str) -> ImageProcessor:
    return ImageProcessor.from_preset(name)


def example():
    processor = ImageProcessor.from_preset("dino")

    images = torch.randint(0, 256, (4, 480, 640, 3), dtype=torch.uint8)
    depths = torch.randint(1, 2000, (4, 480, 640), dtype=torch.int32)
    intrinsics = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_to_world = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5]],
        dtype=torch.float32,
    )

    rgb_out = processor.process_images(images)
    rgbd_out = processor.process_rgbd(images, depths, intrinsics, camera_to_world)

    print("process_images image:", tuple(rgb_out.image.shape))
    print("process_rgbd image   :", tuple(rgbd_out.image.shape))
    print("process_rgbd depth   :", tuple(rgbd_out.depth.shape))
    print("process_rgbd K       :", tuple(rgbd_out.intrinsics.shape))
    print("process_rgbd Tcw     :", tuple(rgbd_out.camera_to_world.shape))


if __name__ == "__main__":
    example()