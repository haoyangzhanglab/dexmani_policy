"""Independent, per-frame XHand tactile encoders.

The 3D force path follows the state-MLP approach of FTP-1 / ReDex.
The 120-taxel path uses the per-finger 1D CNN from KineDex:
https://github.com/DinoMini00/KineDex_code/blob/main/diffusion_policy/model/vision/multi_image_obs_encoder.py

Both paths share their parameters across fingers and produce five 64D tokens.
Leading [B] or [B,T] dimensions are processed independently: no temporal
modeling, cross-finger interaction, or multimodal fusion is performed here.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _prepare_input(
    x: torch.Tensor,
    tactile_valid: torch.Tensor | None,
    input_shape: tuple[int, ...],
    input_key: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate a normalized per-frame input; sanitize masked fingers before encoding."""
    n = len(input_shape)
    if x.ndim not in (n + 1, n + 2) or tuple(x.shape[-n:]) != input_shape or 0 in x.shape[:-n]:
        raise ValueError(f"{input_key}: expected nonempty [B,{input_shape}] or [B,T,{input_shape}], got {tuple(x.shape)}")
    if tactile_valid is not None:
        mask_shape = (*x.shape[:-n], 5)
        if tactile_valid.shape != mask_shape or tactile_valid.dtype != torch.bool:
            raise ValueError(f"tactile_valid must be bool {mask_shape}")
        x = torch.where(tactile_valid.reshape(*mask_shape, *((1,) * (n - 1))), x, 0.0)
    x = x.to(dtype=dtype)
    if not torch.isfinite(x).all():
        raise ValueError(f"{input_key}: nonfinite values in valid observations")
    return x


class XHandForceEncoder(nn.Module):
    """Normalized [B,(T,)5,3] -> [B,(T,)5,64] with one shared 3-layer MLP."""

    num_tokens = 5
    frame_dim = 64
    input_key = "finger_tactile_force"
    input_shape = (5, 3)

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
        x = _prepare_input(x, tactile_valid, self.input_shape, self.input_key, self.net[0].weight.dtype)
        features = self.net(x)
        if tactile_valid is not None:
            features = torch.where(tactile_valid[..., None], features, 0.0)
        return features

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
    input_shape = (5, 120, 3)

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
        x = _prepare_input(x, tactile_valid, self.input_shape, self.input_key, self.net[0].weight.dtype)
        prefix = x.shape[:-3]
        fingers = x.reshape(-1, 120, 3).transpose(1, 2)
        features = self.net(fingers).squeeze(-1).reshape(*prefix, self.num_tokens, self.frame_dim)
        if tactile_valid is not None:
            features = torch.where(tactile_valid[..., None], features, 0.0)
        return features

    @property
    def out_dim(self) -> int:
        return self.frame_dim

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)



# ReTouch Figure 6 + right-hand T30/T16 sensor geometry (reconstructed).
# These are not author-released region_ids and have not been verified to match
# every firmware/SDK raw_force ordering. Index j = 12*r + c (0-based):
#   r=0..9 circumferential strips, c=0..11 base-to-tip.
# IDs: 0 tip, 1 center, 2 base, 3 left, 4 right.
# Sources: https://arxiv.org/html/2608.01824v2 (Figure 6)
#          https://github.com/tsingqingyun/xhand1/tree/main/tactile
_RIGHT_T30_PATCH_ROWS = (
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
_RIGHT_T16_PATCH_ROWS = (
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


def _right_hand_reconstructed_region_ids() -> torch.Tensor:
    """Five [120] labels, thumb T30 then four T16, in assumed SDK index order."""
    rows = (_RIGHT_T30_PATCH_ROWS,) + (_RIGHT_T16_PATCH_ROWS,) * 4
    return torch.tensor(
        [[int(label) for row in finger_rows for label in row] for finger_rows in rows],
        dtype=torch.long,
    )


class XHandReTouchEncoder(nn.Module):
    """ReTouch-inspired five-patch encoder for single-frame XHand taxel arrays.

    By default uses a reconstructed RIGHT-hand (T30 thumb / T16 others)
    region_ids [5,120] inferred from ReTouch Figure 6 and public T30/T16
    coordinates. The paper does not publish its exact SDK taxel-index map:
    the default mapping is evidence-based but NOT an official verified map.
    Pass region_ids explicitly to override it for another hand or SDK order.
    Label convention: 0 tip, 1 center, 2 base, 3 left, 4 right.

    ReTouch Appendix A.2 specifies 8D patch statistics, soft contact weighting,
    8->256->1024 patch MLP and contact-area-biased attention pooling.
    For this repo the MLP ends at the existing 64D token width (8->256->64).
    Auxiliary encoder pretraining from ReTouch is NOT reproduced.

    Input must already be normalized using the training-set statistics.
    The published policy-stage theta=1.0 and gamma=0.5 refer to normalized
    force magnitudes, not Newton units.
    """

    num_tokens = 5
    num_regions = 5
    frame_dim = 64
    input_key = "taxel_tactile_force"
    input_shape = (5, 120, 3)

    def __init__(self, region_ids: torch.Tensor | None = None) -> None:
        super().__init__()
        if region_ids is None:
            region_ids = _right_hand_reconstructed_region_ids()
        labels = torch.as_tensor(region_ids)
        if labels.shape != (5, 120) or labels.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise ValueError("region_ids must be integer [5,120] in SDK taxel index order")
        if not bool(((labels >= 0) & (labels < self.num_regions)).all()):
            raise ValueError("region_ids values must be 0..4 (tip,center,base,left,right)")
        patch_mask = F.one_hot(labels.long(), num_classes=self.num_regions).permute(0, 2, 1).bool()
        if not bool(patch_mask.any(dim=-1).all()):
            raise ValueError("each finger must have at least one taxel in each of five regions")

        # Checkpoints retain the exact taxel-to-patch mapping as a buffer.
        self.register_buffer("patch_mask", patch_mask)
        self.patch_mlp = nn.Sequential(nn.Linear(8, 256), nn.SiLU(), nn.Linear(256, self.frame_dim))
        self.finger_embedding = nn.Parameter(torch.empty(5, self.frame_dim))
        self.region_embedding = nn.Parameter(torch.empty(5, self.frame_dim))
        nn.init.normal_(self.finger_embedding, std=0.02)
        nn.init.normal_(self.region_embedding, std=0.02)
        self.attention_score = nn.Linear(self.frame_dim, 1)
        self.token_norm = nn.LayerNorm(self.frame_dim)

    def forward(self, x: torch.Tensor, tactile_valid: torch.Tensor | None = None) -> torch.Tensor:
        x = _prepare_input(x, tactile_valid, self.input_shape, self.input_key, self.patch_mlp[0].weight.dtype)
        leading = x.shape[:-3]
        # Force norms, contact weights and reductions use float32 under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            f = x.reshape(-1, 5, 120, 3).float()
            magnitude = torch.linalg.vector_norm(f, dim=-1)
            weights = torch.sigmoid((magnitude - 1.0) / 0.5)
            mask = self.patch_mask[None].float()  # [1,5,5,120]
            gated = weights[:, :, None, :] * mask
            mass = gated.sum(dim=-1)
            gated_mean = torch.einsum("bfpt,bftc->bfpc", gated, f) / mass.clamp_min(1e-6)[..., None]
            component_max = (f.abs()[:, :, None, :, :] * mask[..., None]).amax(dim=-2)
            area = mass / mask.sum(dim=-1)
            strength = (magnitude[:, :, None, :] * mask).amax(dim=-1)
            descriptors = torch.cat((gated_mean, component_max, area[..., None], strength[..., None]), dim=-1)

        h = self.patch_mlp(descriptors)
        h = h + self.finger_embedding[None, :, None, :] + self.region_embedding[None, None, :, :]
        # ReTouch Eq. (12): patch-level learned score + log contact-area prior.
        logits = self.attention_score(F.silu(h)).squeeze(-1).float() + torch.log(area + 1e-6)
        alpha = torch.softmax(logits, dim=-1)
        token = self.token_norm((alpha[..., None] * h.float()).sum(dim=2))
        output = token.reshape(*leading, self.num_tokens, self.frame_dim)
        if tactile_valid is not None:
            output = torch.where(tactile_valid[..., None], output, 0.0)
        return output

    @property
    def out_dim(self) -> int:
        return self.frame_dim

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return (self.input_key,)
