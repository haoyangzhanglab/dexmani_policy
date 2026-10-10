"""Same-frame, same-finger tactile residuals followed by hand self-attention."""

import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.pointcloud.scene_compressor import SceneSelfAttentionBlock


class FingerAlignedFusion(nn.Module):
    """Fuse [B,T,6,D] geometry with [B,T,5,F] tactile frame features.

    Geometry slots are wrist, thumb, index, middle, ring, pinky. geometry_meta
    is [B,T,5,7]: wrist-local XYZ, visible distance, observation bit,
    near-support bit, and mean near-null mass (not calibrated uncertainty).
    force_features [B,T,5,3] are normalized sensor channels, not world-frame forces.
    A valid zero reading is informative; only tactile_valid disables injection.

    The output projection starts at zero, so fused_geometry initially equals
    geometry exactly. Compare against a geometry-only model with the same hand
    attention block. The first optimizer step updates the output projection;
    subsequent steps also train its upstream tactile, gain and gate branches.
    """

    def __init__(
        self,
        token_channels: int = 192,
        frame_dim: int = 64,
        num_heads: int = 4,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        if token_channels < 2 or num_heads < 1 or token_channels % num_heads:
            raise ValueError("token_channels must be >= 2 and divisible by num_heads")
        if frame_dim < 1:
            raise ValueError("frame_dim must be positive")
        self.token_channels = token_channels
        self.frame_dim = frame_dim
        self.validate_finite = validate_finite
        self.geometry_norm = nn.LayerNorm(token_channels)
        self.frame_norm = nn.LayerNorm(frame_dim)
        self.tactile_proj = nn.Linear(frame_dim, token_channels, bias=False)
        self.force_proj = nn.Linear(3, token_channels, bias=False)
        self.tactile_norm = nn.LayerNorm(token_channels)
        self.modulation = nn.Sequential(
            nn.Linear(token_channels + 7, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * token_channels + 10, token_channels // 2),
            nn.GELU(),
            nn.Linear(token_channels // 2, 1),
        )
        self.out = nn.Linear(token_channels, token_channels, bias=False)
        self.hand_attention = SceneSelfAttentionBlock(token_channels, num_heads, 2 * token_channels)
        self.output_norm = nn.LayerNorm(token_channels)
        for layer in (self.modulation[-1], self.gate[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.out.weight)

    def forward(
        self,
        geometry: torch.Tensor,
        tactile_frames: torch.Tensor,
        force_features: torch.Tensor,
        geometry_meta: torch.Tensor,
        tactile_valid: torch.Tensor | None = None,
        *,
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        if geometry.ndim != 4 or geometry.shape[2:] != (6, self.token_channels):
            raise ValueError("geometry must have shape [B,T,6,token_channels]")
        batch_size, time_steps = geometry.shape[:2]
        if min(batch_size, time_steps) < 1:
            raise ValueError("batch size and time steps must be positive")
        for value, suffix, name in (
            (geometry, (6, self.token_channels), "geometry"),
            (tactile_frames, (5, self.frame_dim), "tactile_frames"),
            (force_features, (5, 3), "force_features"),
            (geometry_meta, (5, 7), "geometry_meta"),
        ):
            if value.shape != (batch_size, time_steps, *suffix) or value.device != geometry.device:
                raise ValueError(f"{name}: unexpected shape or device")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")
        if tactile_valid is not None:
            if (
                tactile_valid.shape != (batch_size, time_steps, 5)
                or tactile_valid.dtype != torch.bool
                or tactile_valid.device != geometry.device
            ):
                raise ValueError("tactile_valid must be bool [B,T,5] on the geometry device")
            tactile_frames = torch.where(tactile_valid[..., None], tactile_frames, 0.0)
            force_features = torch.where(tactile_valid[..., None], force_features, 0.0)
        if self.validate_finite:
            for value in (geometry, tactile_frames, force_features, geometry_meta):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("fusion inputs must be finite after masking missing tactile")

        dtype = self.geometry_norm.weight.dtype
        g = self.geometry_norm(geometry[:, :, 1:].to(dtype))
        force = force_features.to(dtype)
        meta = geometry_meta.to(dtype)
        evidence = self.tactile_proj(self.frame_norm(tactile_frames.to(dtype)))
        evidence = evidence + self.force_proj(force)
        gain = 1 + 0.5 * self.modulation(torch.cat((g, meta), -1)).tanh()
        gate_input = torch.cat((g, self.tactile_norm(evidence), meta, force), -1)
        alpha = self.gate(gate_input).sigmoid()
        if tactile_valid is not None:
            alpha = alpha * tactile_valid[..., None].to(alpha.dtype)
        update = (alpha * self.out(gain * evidence)).to(geometry.dtype)
        fused = torch.cat((geometry[:, :, :1], geometry[:, :, 1:] + update), dim=2)
        # Only the six hand slots mix here; time remains an explicit dimension.
        hand = self.hand_attention(fused.flatten(0, 1).to(dtype))
        hand = self.output_norm(hand).reshape(batch_size, time_steps, 6, self.token_channels)
        outputs = {"hand_token": hand}
        if return_intermediate:
            outputs.update(
                fused_geometry=fused,
                touch_gate=alpha.squeeze(-1),
                touch_update=update,
                channel_gain=gain,
            )
        return outputs

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (6, self.token_channels)
