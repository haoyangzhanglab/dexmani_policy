"""Gated tactile fusion for hand-scene tokens."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..proprio import HandStateEncoder
from ..xhand_tactile import XHandForceEncoder


class GatedTactileFusion(nn.Module):
    KEYPOINTS = HandStateEncoder.KEYPOINTS

    def __init__(
        self,
        token_channels: int = 192,
        *,
        tactile_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.tactile_encoder = XHandForceEncoder() if tactile_encoder is None else tactile_encoder
        self.input_shape = tuple(self.tactile_encoder.input_shape)
        self.token_channels = token_channels

        tactile_dim = self.tactile_encoder.out_dim
        self.num_heads = 4
        self.head_dim = tactile_dim // self.num_heads

        self.finger_embedding = nn.Parameter(torch.empty(1, 5, tactile_dim))
        self.summary_token = nn.Parameter(torch.empty(1, 1, tactile_dim))
        nn.init.normal_(self.finger_embedding, std=0.02)
        nn.init.normal_(self.summary_token, std=0.02)
        self.attn_norm = nn.LayerNorm(tactile_dim)
        self.self_attn = nn.MultiheadAttention(
            tactile_dim, self.num_heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(tactile_dim)
        self.ffn = nn.Sequential(
            nn.Linear(tactile_dim, 2 * tactile_dim), nn.GELU(), nn.Linear(2 * tactile_dim, tactile_dim)
        )

        self.query_norm = nn.LayerNorm(token_channels)
        self.tactile_norm = nn.LayerNorm(tactile_dim)
        self.summary_norm = nn.LayerNorm(tactile_dim)
        self.q = nn.Linear(token_channels, tactile_dim)
        self.kv = nn.Linear(tactile_dim, 2 * tactile_dim)
        self.proj = nn.Linear(tactile_dim, token_channels)
        self.gate_mlp = nn.Sequential(
            nn.Linear(token_channels + tactile_dim + 4 + 5, tactile_dim),
            nn.SiLU(),
            nn.Linear(tactile_dim, 1),
        )

        # Nonzero residual weights let tactile and gate paths learn from step one.
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, math.log(0.1 / 0.9))

    def _encode_context(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch = tokens.shape[0]
        context = torch.cat(
            (tokens.float() + self.finger_embedding.float(),
             self.summary_token.float().expand(batch, -1, -1)),
            1,
        )
        context_valid = torch.cat((valid, valid.new_ones(batch, 1)), -1)
        normalized = self.attn_norm(context.to(self.attn_norm.weight.dtype))
        context = context + self.self_attn(
            normalized, normalized, normalized, key_padding_mask=~context_valid, need_weights=False
        )[0].float()
        context = context + self.ffn(self.ffn_norm(context.to(self.ffn_norm.weight.dtype))).float()
        return context[:, :5], context[:, 5]

    def forward(
        self,
        hand_scene: Mapping[str, Tensor],
        *,
        tactile: Tensor | None = None,
        tactile_valid: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> dict[str, Tensor]:
        """Return complete [N,6,D] tokens; callers must not add hand or scene residuals again."""
        # FP32 residuals preserve small tactile updates under BF16 autocast.
        x = hand_scene["hand_query"].float() + hand_scene["relation_update"].float()
        batch = x.shape[0]
        if tactile is None:
            if tactile_valid is None:
                tactile_valid = torch.zeros(batch, 5, dtype=torch.bool, device=x.device)
            elif tactile_valid.any():
                raise ValueError("tactile=None cannot have valid fingers")
            tactile = x.new_zeros(batch, *self.input_shape)

        tokens = self.tactile_encoder(tactile, tactile_valid=tactile_valid)
        if tactile_valid is None:
            tactile_valid = torch.ones(batch, 5, dtype=torch.bool, device=x.device)
        local, summary = self._encode_context(tokens, tactile_valid)
        available = tactile_valid.any(-1, keepdim=True)

        query = self.query_norm(x.to(self.query_norm.weight.dtype))
        memory = self.tactile_norm(local.to(self.tactile_norm.weight.dtype))
        q = self.q(query).reshape(batch, 6, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(memory)

        # SDPA True means allowed; only empty frames can read the zero K/V slot.
        kv = F.pad(kv, (0, 0, 0, 1))
        k, v = kv.reshape(batch, 6, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        allowed = torch.cat((tactile_valid, ~available), -1)
        delta = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None, None])
        delta = delta.transpose(1, 2).reshape(batch, 6, -1)
        delta = torch.where(available[..., None], self.proj(delta).float(), 0.0)

        normalized_summary = self.summary_norm(summary.to(self.summary_norm.weight.dtype))
        gate_input = torch.cat(
            (query, normalized_summary[:, None].expand(-1, 6, -1),
             hand_scene["geometry_descriptor"][..., 3:].float(),
             tactile_valid[:, None].float().expand(-1, 6, -1)),
            -1,
        )
        gate = self.gate_mlp(gate_input.to(self.gate_mlp[0].weight.dtype)).float().sigmoid()
        gate = gate * available[..., None]
        tactile_update = gate * delta

        output = {"interaction_tokens": x + tactile_update}
        if return_intermediate:
            output.update(
                tactile_tokens=local,
                tactile_summary=summary,
                tactile_valid=tactile_valid,
                gate=gate.squeeze(-1),
                tactile_update=tactile_update,
            )
        return output

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return 6, self.token_channels
