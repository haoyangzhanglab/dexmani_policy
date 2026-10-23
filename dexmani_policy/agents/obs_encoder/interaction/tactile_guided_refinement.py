"""Tactile-Guided Refinement (TGR) of hand-scene interaction relations."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..proprio import HandSpatialEncoder
from ..xhand_tactile import XHandForceEncoder


def normalize_evidence(tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Separate latent direction and log magnitude before multimodal conditioning."""
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        tokens = tokens.float()
        magnitude = (tokens.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        return tokens / magnitude, magnitude.log1p()


class TactileGuidedRefinement(nn.Module):
    """Refine geometric interaction hypotheses with local and whole-grasp evidence.

    Call after HandSceneEncoder with its ``relation_update`` [BT,6,D] and
    ``geometry_descriptor`` [BT,6,7]. Geometry slots follow
    wrist/thumb/index/middle/ring/pinky; tactile slots follow the five fingers
    of the same hand and frame. BT is the flattened observation batch and time.

    Local evidence pairs each relation with its own tactile feature. Grasp
    evidence combines the five ordered pairs with wrist geometry. Both use a
    shared gated affine operator, r' = r + m * sigmoid(a) * candidate, where
    candidate = refinement_scale * (tanh(gamma) * RMSNorm(r) + tanh(beta)).
    Each relation gate controls only its own residual, after all normalization.
    A finger's evidence can inform the grasp even when its local gate is closed.

    ``tactile_valid`` masks sensor availability in every path. It is not a
    contact label: valid zero readings remain evidence, and visual proximity
    never hard-disables touch. Missing fingers keep their original relations;
    a wholly missing tactile observation also leaves the wrist unchanged.
    Gates are task-trained refinement weights, not contact probabilities.

    Reuses XHandForceEncoder (default), XHandTaxelEncoder or XHandPatchEncoder.
    Inputs must use that encoder's normalization. No force-to-scene coordinate
    registration is assumed. Coordinates and geometric diagnostics are preserved;
    only latent relations change. Temporal reasoning stays with the caller.
    Small nonzero affine weights and conservative gates allow first-step gradients
    from the existing action objective without additional labels or losses.

    Example::

        # Construct once alongside the policy's other encoders.
        refinement = TactileGuidedRefinement(token_channels=hand_scene_encoder.out_dim)

        scene = hand_scene_encoder(points, patches, anchors, hand_tokens)
        refined = refinement(scene, tactile=normalized_force, tactile_valid=valid)
        hand_features = hand_tokens + refined["relation_update"]  # Apply once.
    """

    KEYPOINTS = HandSpatialEncoder.KEYPOINTS

    def __init__(
        self,
        token_channels: int = 192,
        *,
        tactile_encoder: nn.Module | None = None,
        hidden_dim: int = 64,
        refinement_scale: float = 0.1,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (token_channels, hidden_dim)):
            raise ValueError("token_channels and hidden_dim must be positive integers")
        if not math.isfinite(refinement_scale) or refinement_scale < 0:
            raise ValueError("refinement_scale must be finite and nonnegative")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be strictly between zero and one")
        self.tactile_encoder = XHandForceEncoder() if tactile_encoder is None else tactile_encoder
        if getattr(self.tactile_encoder, "num_tokens", None) != 5:
            raise ValueError("tactile_encoder must implement the five-finger XHand encoder interface")
        self.input_shape = tuple(self.tactile_encoder.input_shape)
        if self.input_shape[0] != 5:
            raise ValueError("tactile_encoder.input_shape must start with five fingers")

        self.token_channels = token_channels
        self.refinement_scale = float(refinement_scale)
        local_dim = token_channels + self.tactile_encoder.out_dim + 7 + 2
        grasp_dim = token_channels + 7 + 1 + 5 * (hidden_dim + 1)
        self.local_evidence = nn.Sequential(nn.Linear(local_dim, hidden_dim), nn.SiLU())
        self.grasp_evidence = nn.Sequential(nn.Linear(grasp_dim, hidden_dim), nn.SiLU())
        self.refinement_gate = nn.Linear(hidden_dim, 1)
        self.affine_modulation = nn.Linear(hidden_dim, 2 * token_channels)

        # Small nonzero residuals let the gate and evidence encoders learn from step one.
        nn.init.normal_(self.affine_modulation.weight, std=0.01)
        nn.init.zeros_(self.affine_modulation.bias)
        nn.init.normal_(self.refinement_gate.weight, std=0.01)
        nn.init.constant_(self.refinement_gate.bias, math.log(gate_init / (1 - gate_init)))

    def encode_tactile(
        self, tactile: Tensor | None, tactile_valid: Tensor | None, batch: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Mask missing values before and after encoding, without skipping trainable paths."""
        if tactile_valid is not None:
            if tactile_valid.shape != (batch, 5) or tactile_valid.dtype != torch.bool:
                raise ValueError("tactile_valid must be bool [BT,5]")
            if tactile_valid.device != device:
                raise ValueError("tactile_valid and geometry must be on the same device")
        if tactile is None:
            if tactile_valid is not None and bool(tactile_valid.any()):
                raise ValueError("tactile=None cannot have valid fingers")
            tactile_valid = torch.zeros(batch, 5, dtype=torch.bool, device=device)
            tactile = torch.zeros(batch, *self.input_shape, device=device)
        else:
            if tactile.shape != (batch, *self.input_shape) or not tactile.is_floating_point():
                raise ValueError(f"tactile must be floating point [BT,{self.input_shape}]")
            if tactile.device != device:
                raise ValueError("tactile and geometry must be on the same device")
            if tactile_valid is None:
                tactile_valid = torch.ones(batch, 5, dtype=torch.bool, device=device)

        mask = tactile_valid.reshape(batch, 5, *((1,) * (len(self.input_shape) - 1)))
        tactile = torch.where(mask, tactile, 0.0)
        if not bool(torch.isfinite(tactile).all()):
            raise ValueError("valid tactile values must be finite; mask unavailable fingers explicitly")
        tokens = self.tactile_encoder(tactile, tactile_valid=tactile_valid)
        if tokens.shape != (batch, 5, self.tactile_encoder.out_dim):
            raise ValueError("tactile_encoder must return [BT,5,out_dim]")
        return torch.where(tactile_valid[..., None], tokens, 0.0), tactile_valid

    def refine_relations(
        self, relations: Tensor, normalized: Tensor, evidence: Tensor, available: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply one bounded affine residual and one gate per relation."""
        gate = self.refinement_gate(evidence).float().sigmoid()
        affine = self.affine_modulation(evidence).float().tanh()
        scale, shift = affine.chunk(2, -1)
        gate = torch.where(available[..., None], gate, 0.0)
        residual = self.refinement_scale * gate * (scale * normalized + shift)
        return relations + residual.to(relations.dtype), gate.squeeze(-1), residual

    def forward(
        self,
        hand_scene: Mapping[str, Tensor],
        *,
        tactile: Tensor | None = None,
        tactile_valid: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        """Return new relations, optionally exposing gates and the FP32 residual before casting."""
        geometry, descriptor = hand_scene["relation_update"], hand_scene["geometry_descriptor"]
        if geometry.ndim != 3 or geometry.shape[1:] != (6, self.token_channels) or geometry.shape[0] < 1:
            raise ValueError("relation_update must be [BT,6,token_channels] with a nonempty batch")
        if descriptor.shape != (*geometry.shape[:2], 7) or descriptor.device != geometry.device:
            raise ValueError("geometry_descriptor must be [BT,6,7] on the relation device")
        if any(not x.is_floating_point() or not bool(torch.isfinite(x).all()) for x in (geometry, descriptor)):
            raise ValueError("relation_update and geometry_descriptor must be finite floating-point tensors")

        tactile_tokens, present = self.encode_tactile(tactile, tactile_valid, geometry.shape[0], geometry.device)
        normalized, geometry_magnitude = normalize_evidence(geometry)
        tactile_direction, tactile_magnitude = normalize_evidence(tactile_tokens)
        local_input = torch.cat(
            (
                normalized[:, 1:],
                tactile_direction,
                descriptor[:, 1:].float(),
                geometry_magnitude[:, 1:],
                tactile_magnitude,
            ),
            -1,
        )
        local = self.local_evidence(local_input.to(self.local_evidence[0].weight.dtype))
        local = torch.where(present[..., None], local, 0.0)

        # Ordered slots preserve finger identity; presence distinguishes missing from valid zero touch.
        grasp_context = torch.cat((local.float(), present[..., None].float()), -1).flatten(1)
        grasp_input = torch.cat(
            (normalized[:, 0], descriptor[:, 0].float(), geometry_magnitude[:, 0], grasp_context), -1
        )
        grasp = self.grasp_evidence(grasp_input.to(self.grasp_evidence[0].weight.dtype))
        evidence = torch.cat((grasp[:, None], local), 1)
        available = torch.cat((present.any(-1, keepdim=True), present), -1)
        relation, gate, residual = self.refine_relations(geometry, normalized, evidence, available)
        output = dict(hand_scene)
        output.update(relation_update=relation, wrist_relation=relation[:, :1], finger_relation=relation[:, 1:])
        if return_intermediate:
            output.update(
                geometry_relation=geometry,
                tactile_tokens=tactile_tokens,
                tactile_present=present,
                refinement_gate=gate,
                refinement_update=residual,
            )
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return 6, self.token_channels
