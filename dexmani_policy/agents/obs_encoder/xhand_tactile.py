"""Per-frame XHand force, taxel and regional patch encoders."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.common.config import FIELD_FEATURE_SHAPES

__all__ = ["XHandForceEncoder", "XHandTaxelEncoder", "XHandPatchEncoder"]


def prepare_tactile_input(
    x: torch.Tensor,
    tactile_valid: torch.Tensor | None,
    input_shape: tuple[int, ...],
    input_key: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mask missing fingers before encoding; valid inputs must be finite and normalized."""
    feature_dims = len(input_shape)
    if x.ndim not in (feature_dims + 1, feature_dims + 2) or x.shape[-feature_dims:] != input_shape:
        raise ValueError(
            f"{input_key}: expected [B,{input_shape}] or [B,T,{input_shape}], got {tuple(x.shape)}"
        )

    if tactile_valid is not None:
        mask_shape = (*x.shape[:-feature_dims], input_shape[0])
        if tactile_valid.shape != mask_shape or tactile_valid.dtype != torch.bool:
            raise ValueError(f"tactile_valid must be bool {mask_shape}")
        x = torch.where(tactile_valid.reshape(*mask_shape, *((1,) * (feature_dims - 1))), x, 0.0)

    return x.to(dtype=dtype)


class XHandForceEncoder(nn.Module):
    """Normalized [B,(T,)5,3] -> [B,(T,)5,64] with one shared 3-layer MLP."""

    num_tokens = 5
    frame_dim = 64
    input_key = "finger_tactile_force"
    input_shape = FIELD_FEATURE_SHAPES[input_key]

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, x: torch.Tensor, tactile_valid: torch.Tensor | None = None) -> torch.Tensor:
        x = prepare_tactile_input(
            x, tactile_valid, self.input_shape, self.input_key, self.net[0].weight.dtype
        )
        features = self.net(x)

        if tactile_valid is not None:
            features = torch.where(tactile_valid[..., None], features, 0.0)
        return features

    @property
    def effective_config(self):
        return {"_target_": f"{type(self).__module__}.{type(self).__name__}"}

    @property
    def out_dim(self) -> int:
        return self.frame_dim

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)


class XHandTaxelEncoder(nn.Module):
    """Normalized [B,(T,)5,120,3] -> [B,(T,)5,64] with KineDex 1D CNN."""

    num_tokens = 5
    frame_dim = 64
    input_key = "taxel_tactile_force"
    input_shape = FIELD_FEATURE_SHAPES[input_key]

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor, tactile_valid: torch.Tensor | None = None) -> torch.Tensor:
        x = prepare_tactile_input(
            x, tactile_valid, self.input_shape, self.input_key, self.net[0].weight.dtype
        )
        batch_shape = x.shape[:-3]
        fingers = x.reshape(-1, 120, 3).transpose(1, 2)
        features = (
            self.net(fingers).squeeze(-1).reshape(*batch_shape, self.num_tokens, self.frame_dim)
        )

        if tactile_valid is not None:
            features = torch.where(tactile_valid[..., None], features, 0.0)
        return features

    @property
    def effective_config(self):
        return {"_target_": f"{type(self).__module__}.{type(self).__name__}"}

    @property
    def out_dim(self) -> int:
        return self.frame_dim

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)


# Reconstructed right-hand regions; these are not a verified SDK map.
RIGHT_T30_PATCH_ROWS = (
    "233333333333",
    "333333333333",
    "223333333300",
    "222111110000",
    "222111110000",
    "222111110000",
    "222111111000",
    "224444444400",
    "444444444444",
    "244444444444",
)
RIGHT_T16_PATCH_ROWS = (
    "233333333330",
    "333333333300",
    "333333333000",
    "222113330000",
    "222111110000",
    "222111110000",
    "222111140000",
    "244444444000",
    "444444444400",
    "244444444440",
)


def right_hand_region_ids() -> torch.Tensor:
    """Reconstructed [5,120] labels: tip=0, center=1, base=2, left=3, right=4."""
    rows = (RIGHT_T30_PATCH_ROWS,) + (RIGHT_T16_PATCH_ROWS,) * 4
    return torch.tensor(
        [[int(label) for row in finger_rows for label in row] for finger_rows in rows],
        dtype=torch.long,
    )


class XHandPatchEncoder(nn.Module):
    """Regional force descriptors and soft-activity pooling into five 64D tokens."""

    num_tokens = 5
    num_regions = 5
    frame_dim = 64
    input_key = "taxel_tactile_force"
    input_shape = FIELD_FEATURE_SHAPES[input_key]

    def __init__(self, region_ids: torch.Tensor | None = None, activity_scale: float = 1.0) -> None:
        super().__init__()
        activity_scale = torch.tensor(activity_scale, dtype=torch.float32).item()
        if not math.isfinite(activity_scale) or activity_scale <= 0:
            raise ValueError("activity_scale must be finite and positive in FP32")
        self.activity_scale = activity_scale
        if region_ids is None:
            region_ids = right_hand_region_ids()
        labels = torch.as_tensor(region_ids)
        if labels.shape != (5, 120) or labels.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("region_ids must be integer [5,120] in SDK taxel index order")
        if (labels < 0).any() or (labels >= self.num_regions).any():
            raise ValueError("region_ids must be in [0, 4]")
        labels = labels.to(dtype=torch.long, copy=True)
        region_sizes = labels.new_zeros(5, self.num_regions).scatter_add_(1, labels, torch.ones_like(labels))
        if not bool((region_sizes > 0).all()):
            raise ValueError("each finger must have at least one taxel in each of five regions")

        # Rebuilt from effective_config.region_ids; no duplicate checkpoint state.
        self.register_buffer("region_ids", labels, persistent=False)
        self.register_buffer("region_sizes", region_sizes, persistent=False)
        self.patch_mlp = nn.Sequential(nn.Linear(8, 256), nn.SiLU(), nn.Linear(256, self.frame_dim))
        self.finger_embedding = nn.Parameter(torch.empty(5, self.frame_dim))
        self.region_embedding = nn.Parameter(torch.empty(5, self.frame_dim))
        nn.init.normal_(self.finger_embedding, std=0.02)
        nn.init.normal_(self.region_embedding, std=0.02)
        self.attention_score = nn.Linear(self.frame_dim, 1)
        self.token_norm = nn.LayerNorm(self.frame_dim)

    @property
    def effective_config(self):
        return {
            "_target_": f"{type(self).__module__}.{type(self).__name__}",
            "activity_scale": self.activity_scale,
            "region_ids": self.region_ids.tolist(),
        }

    def _patch_descriptors(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):
            force = z.reshape(-1, 5, 120, 3).float()
            magnitude = torch.linalg.vector_norm(force, dim=-1)
            activity = magnitude / (magnitude + self.activity_scale)
            region = self.region_ids[None].expand(force.shape[0], -1, -1)
            shape = (force.shape[0], 5, self.num_regions)
            mass = activity.new_zeros(shape).scatter_add_(2, region, activity)
            denominator = torch.where(mass > 0, mass, 1.0)
            force_region = region[..., None].expand_as(force)
            mean_force = force.new_zeros(*shape, 3).scatter_add_(
                2, force_region, force * activity[..., None],
            ) / denominator[..., None]
            peak_force = force.new_zeros(*shape, 3).scatter_reduce_(
                2, force_region, force.abs(), reduce="amax", include_self=False,
            )
            activity_fraction = mass / self.region_sizes
            strength = magnitude.new_zeros(shape).scatter_reduce_(
                2, region, magnitude, reduce="amax", include_self=False,
            )
            return torch.cat(
                (mean_force, peak_force, activity_fraction[..., None], strength[..., None]), dim=-1
            )

    def forward(self, x: torch.Tensor, tactile_valid: torch.Tensor | None = None) -> torch.Tensor:
        x = prepare_tactile_input(x, tactile_valid, self.input_shape, self.input_key, torch.float32)
        batch_shape = x.shape[:-3]
        descriptors = self._patch_descriptors(x)
        patch_features = self.patch_mlp(descriptors)
        patch_features = (
            patch_features
            + self.finger_embedding[None, :, None, :]
            + self.region_embedding[None, None, :, :]
        )
        # Soft activity biases pooling; it is not a calibrated contact probability.
        logits = self.attention_score(F.silu(patch_features)).squeeze(-1).float() + torch.log(
            descriptors[..., 6] + 1e-6
        )
        attention_weights = torch.softmax(logits, dim=-1)
        tokens = self.token_norm((attention_weights[..., None] * patch_features.float()).sum(dim=2))
        features = tokens.reshape(*batch_shape, self.num_tokens, self.frame_dim)

        if tactile_valid is not None:
            features = torch.where(tactile_valid[..., None], features, 0.0)
        return features

    @property
    def out_dim(self) -> int:
        return self.frame_dim

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)
