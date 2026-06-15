import os
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from typing import Dict, Literal, Optional, Sequence

from dexmani_policy.agents.obs_encoder.rgb.common.image_processor import ImageProcessor
from dexmani_policy.agents.obs_encoder.rgb.common.geometry_processor import GeometryProcessor
from dexmani_policy.agents.obs_encoder.rgb.common.types import NormMode
from dexmani_policy.agents.obs_encoder.rgb.common.utils import (
    flatten_batch,
    restore_batch,
    reshape_patch_tokens_to_map,
)
from dexmani_policy.agents.obs_encoder.rgb.resnet import (
    FrozenBatchNorm2d,
    replace_batch_norm_with_group_norm,
)

logger = logging.getLogger(__name__)

TuneMode = Literal["freeze", "full"]
GlobalTokenType = Literal["avg"]

# Google Drive URLs for R3M pretrained checkpoints.
_R3M_URLS = {
    "resnet18": "https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-",
    "resnet34": "https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE",
    "resnet50": "https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA",
}

_R3M_HIDDEN_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

_R3M_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_R3M_IMAGENET_STD = (0.229, 0.224, 0.225)


def _download_r3m_checkpoint(model_name: str) -> str:
    """Download the R3M checkpoint to ``~/.r3m/`` if not already cached.

    Returns the path to the downloaded checkpoint file.
    """
    if model_name not in _R3M_URLS:
        raise ValueError(
            f"Unsupported R3M model: {model_name}. "
            f"Choose from: {sorted(_R3M_URLS.keys())}"
        )

    home = os.path.join(os.path.expanduser("~"), ".r3m")
    size = model_name.replace("resnet", "")
    folder = os.path.join(home, f"r3m_{size}")
    ckpt_path = os.path.join(folder, "model.pt")

    if os.path.exists(ckpt_path):
        return ckpt_path

    os.makedirs(folder, exist_ok=True)

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download R3M weights. "
            "Install it with: pip install gdown"
        )

    logger.info("Downloading R3M %s weights to %s ...", model_name, ckpt_path)
    gdown.download(_R3M_URLS[model_name], ckpt_path, quiet=False)
    return ckpt_path


def _load_r3m_convnet_state_dict(model_name: str) -> Dict[str, torch.Tensor]:
    """Load R3M checkpoint and extract the ResNet convnet state dict.

    The R3M checkpoint was saved from a ``DataParallel``-wrapped model, so
    convnet keys are prefixed ``module.convnet.``.  We strip the prefix to
    produce keys matching a standard ``torchvision.models.resnet*``.
    """
    ckpt_path = _download_r3m_checkpoint(model_name)
    full_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "r3m" not in full_state:
        raise KeyError(
            f"R3M checkpoint at {ckpt_path} does not contain the expected 'r3m' key. "
            f"Found keys: {sorted(full_state.keys())}"
        )

    r3m_state = full_state["r3m"]
    prefix = "module.convnet."
    convnet_state = {
        k[len(prefix):]: v
        for k, v in r3m_state.items()
        if k.startswith(prefix)
    }

    if not convnet_state:
        raise RuntimeError(
            f"No convnet keys found in R3M checkpoint {ckpt_path}. "
            f"Expected keys with prefix '{prefix}'."
        )

    return convnet_state


class R3M(nn.Module):
    """R3M visual backbone — ResNet with Ego4D-pretrained weights.

    R3M (Reusable Representations for Robotic Manipulation) provides ResNet
    weights trained on egocentric video with time-contrastive learning.  This
    class extracts the convnet weights from the official R3M checkpoint and
    wraps them in the same interface as :class:`ResNet`.

    Parameters
    ----------
    model_name:
        One of ``"resnet18"``, ``"resnet34"``, ``"resnet50"``.
    tune_mode:
        ``"freeze"`` — backbone is frozen in eval mode.
        ``"full"``   — all parameters are trainable.
    global_token_type:
        How to pool the spatial feature map.  Currently only ``"avg"``.
    out_dim:
        Optional output dimension.  If ``None``, uses the backbone's native
        hidden dimension (512 for resnet18/34, 2048 for resnet50).
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        tune_mode: TuneMode = "freeze",
        norm_mode: NormMode = "group_norm",
        global_token_type: GlobalTokenType = "avg",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        if model_name not in _R3M_HIDDEN_DIM:
            raise ValueError(
                f"Unsupported R3M model_name: {model_name}. "
                f"Choose from: {sorted(_R3M_HIDDEN_DIM.keys())}"
            )

        self.model_name = model_name
        self.tune_mode = tune_mode
        self.norm_mode = norm_mode
        self.global_token_type = global_token_type
        self.output_stride = 32

        # ── Build a standard ResNet and load R3M weights ──
        resnet_fn = getattr(torchvision.models, model_name)
        norm_layer = FrozenBatchNorm2d if norm_mode == "frozen_bn" else nn.BatchNorm2d
        backbone = resnet_fn(weights=None, norm_layer=norm_layer)
        backbone.fc = nn.Identity()

        convnet_state = _load_r3m_convnet_state_dict(model_name)
        missing, unexpected = backbone.load_state_dict(convnet_state, strict=False)
        if missing:
            logger.warning("R3M %s: %d missing keys (expected fc.*): %s",
                           model_name, len(missing), missing[:5])
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in R3M {model_name} state dict: {unexpected}"
            )

        if norm_mode == "group_norm":
            backbone = replace_batch_norm_with_group_norm(backbone)

        self.hidden_dim = _R3M_HIDDEN_DIM[model_name]
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        # Strip avgpool + fc so the backbone outputs a spatial feature map,
        # matching the existing ResNet backbone convention.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.proj = (
            nn.Identity()
            if self.out_dim == self.hidden_dim
            else nn.Conv2d(self.hidden_dim, self.out_dim, kernel_size=1)
        )

        # R3M was trained with inputs in [0, 255] followed by ImageNet
        # normalization.  The ImageProcessor passes [0, 1] images (mean=0,
        # std=1 preset), so we convert back to [0, 255] and normalise here.
        self.normlayer = T.Normalize(
            mean=list(_R3M_IMAGENET_MEAN),
            std=list(_R3M_IMAGENET_STD),
        )

        self.geometry_processor = GeometryProcessor()

        self.set_tune_mode(tune_mode)

    # ------------------------------------------------------------------
    # Tune mode
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_global_token(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.global_token_type == "avg":
            return feature_map.mean(dim=(-2, -1))

        raise ValueError(f"Unsupported global_token_type: {self.global_token_type}")

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract visual features.

        Parameters
        ----------
        rgb:
            Float tensor in **[0, 1]** with shape ``[..., 3, H, W]``.
            The ImageProcessor preset ``"r3m"`` passes images through
            unchanged (mean=0, std=1), so the values stay in [0, 1].

        Returns
        -------
        dict with keys ``"patch_tokens"`` and ``"global_token"``.
        """
        if rgb.ndim < 4 or rgb.shape[-3] != 3:
            raise ValueError(
                f"rgb should have shape [..., 3, H, W], got {tuple(rgb.shape)}"
            )

        if self.tune_mode == "freeze":
            self.backbone.eval()

        flat_rgb, leading_shape = flatten_batch(rgb, trailing_ndim=3)

        # Match R3M's training preprocessing: [0, 1] → [0, 255] → ImageNet norm.
        flat_rgb = flat_rgb.mul(255.0)
        flat_rgb = self.normlayer(flat_rgb)

        feature_map = self.backbone(flat_rgb)
        feature_map = self.proj(feature_map)

        patch_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        global_token = self.get_global_token(feature_map)

        return {
            "patch_tokens": restore_batch(patch_tokens, leading_shape),
            "global_token": restore_batch(global_token, leading_shape),
        }

    # ------------------------------------------------------------------
    # Geometry (RGB-D support — same pattern as ResNet)
    # ------------------------------------------------------------------

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

    def patch_tokens_to_featmap(
        self, patch_tokens: torch.Tensor, image_hw: Sequence[int]
    ) -> torch.Tensor:
        feature_h = (int(image_hw[0]) + self.output_stride - 1) // self.output_stride
        feature_w = (int(image_hw[1]) + self.output_stride - 1) // self.output_stride

        flat_patch_tokens, leading_shape = flatten_batch(
            patch_tokens, trailing_ndim=2
        )
        feature_map = reshape_patch_tokens_to_map(
            flat_patch_tokens, (feature_h, feature_w)
        )
        return restore_batch(feature_map, leading_shape)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

def example() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "resnet18"

    image_processor = ImageProcessor.from_preset("r3m")

    images = torch.randint(0, 256, (2, 4, 480, 640, 3), dtype=torch.uint8)

    try:
        encoder = R3M(model_name=model_name, tune_mode="freeze").to(device)
        encoder.eval()

        rgb_out = image_processor.process_images(images)
        rgb = rgb_out["image"].to(device)

        with torch.no_grad():
            vision_out = encoder(rgb)
            feature_map = encoder.patch_tokens_to_featmap(
                vision_out["patch_tokens"],
                image_hw=rgb.shape[-2:],
            )

        print("rgb             :", tuple(rgb.shape))
        print("patch_tokens    :", tuple(vision_out["patch_tokens"].shape))
        print("global_token    :", tuple(vision_out["global_token"].shape))
        print("feature_map     :", tuple(feature_map.shape))
        print("out_dim         :", encoder.out_dim)
        print("hidden_dim      :", encoder.hidden_dim)
        print("=== R3M example PASSED ===")

    except Exception as error:
        print("R3M example failed.")
        print(error)
        raise


if __name__ == "__main__":
    example()
