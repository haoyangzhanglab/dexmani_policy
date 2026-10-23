"""Tactile-Guided Refinement (TGR) of hand-scene interaction relations."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..proprio import HandSpatialEncoder
from ..xhand_tactile import XHandForceEncoder


def decompose_evidence(tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Separate RMS-normalized direction and log magnitude in FP32."""
    tokens = tokens.float()
    magnitude = (tokens.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    return tokens / magnitude, magnitude.log1p()


class TactileGuidedRefinement(nn.Module):
    """Refine wrist–finger relations with local tactile and whole-grasp evidence."""

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
        if min(token_channels, hidden_dim) < 1:
            raise ValueError("token_channels and hidden_dim must be positive")
        if not math.isfinite(refinement_scale) or refinement_scale < 0:
            raise ValueError("refinement_scale must be finite and nonnegative")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be strictly between zero and one")

        self.tactile_encoder = XHandForceEncoder() if tactile_encoder is None else tactile_encoder
        self.input_shape = tuple(self.tactile_encoder.input_shape)
        if self.input_shape[0] != 5:
            raise ValueError(
                "tactile_encoder must implement the five-finger XHand encoder interface"
            )

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
        """Delegate masking to the XHand encoder and keep missing-touch paths in autograd."""
        if tactile is None:
            if tactile_valid is None:
                tactile_valid = torch.zeros(batch, 5, dtype=torch.bool, device=device)
            elif bool(tactile_valid.any()):
                raise ValueError("tactile=None cannot have valid fingers")
            tactile = torch.zeros(batch, *self.input_shape, device=device)

        tokens = self.tactile_encoder(tactile, tactile_valid=tactile_valid)
        if tokens.shape != (batch, 5, self.tactile_encoder.out_dim):
            raise ValueError("tactile_encoder must return [BT,5,out_dim]")
        if tactile_valid is None:
            tactile_valid = torch.ones(batch, 5, dtype=torch.bool, device=device)
        return tokens, tactile_valid

    def modulate_relations(
        self, relations: Tensor, direction: Tensor, evidence: Tensor, available: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply one bounded affine residual and one gate per relation."""
        gate = self.refinement_gate(evidence).float().sigmoid()
        affine = self.affine_modulation(evidence).float().tanh()
        scale, shift = affine.chunk(2, -1)
        gate = torch.where(available[..., None], gate, 0.0)
        residual = self.refinement_scale * gate * (scale * direction + shift)
        return relations + residual.to(relations.dtype), gate.squeeze(-1), residual

    def forward(
        self,
        hand_scene: Mapping[str, Tensor],
        *,
        tactile: Tensor | None = None,
        tactile_valid: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        """Refine finite HandSceneEncoder outputs using finite, encoder-normalized tactile observations."""
        relations = hand_scene["relation_update"]
        descriptor = hand_scene["geometry_descriptor"]
        if relations.ndim != 3 or relations.shape[1:] != (6, self.token_channels):
            raise ValueError("relation_update must be [BT,6,token_channels] in KEYPOINTS order")
        if not relations.is_floating_point():
            raise ValueError("relation_update must be floating point to preserve residual updates")
        if descriptor.shape != (*relations.shape[:2], 7):
            raise ValueError("geometry_descriptor must be [BT,6,7]")
        descriptor = descriptor.float()

        tactile_tokens, present = self.encode_tactile(
            tactile, tactile_valid, relations.shape[0], relations.device
        )
        relation_direction, relation_magnitude = decompose_evidence(relations)
        tactile_direction, tactile_magnitude = decompose_evidence(tactile_tokens)
        local_input = torch.cat(
            (
                relation_direction[:, 1:],
                tactile_direction,
                descriptor[:, 1:],
                relation_magnitude[:, 1:],
                tactile_magnitude,
            ),
            -1,
        )
        local_evidence = self.local_evidence(local_input.to(self.local_evidence[0].weight.dtype))
        local_evidence = torch.where(present[..., None], local_evidence, 0.0)

        # Ordered slots preserve finger identity; presence distinguishes missing from valid zero touch.
        grasp_context = torch.cat((local_evidence.float(), present[..., None]), -1).flatten(1)
        grasp_input = torch.cat(
            (relation_direction[:, 0], descriptor[:, 0], relation_magnitude[:, 0], grasp_context),
            -1,
        )
        grasp_evidence = self.grasp_evidence(grasp_input.to(self.grasp_evidence[0].weight.dtype))

        evidence = torch.cat((grasp_evidence[:, None], local_evidence), 1)
        available = torch.cat((present.any(-1, keepdim=True), present), -1)
        refined, gate, residual = self.modulate_relations(
            relations, relation_direction, evidence, available
        )

        output = dict(hand_scene)
        output.update(
            relation_update=refined, wrist_relation=refined[:, :1], finger_relation=refined[:, 1:]
        )
        if return_intermediate:
            output.update(
                geometry_relation=relations,
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
