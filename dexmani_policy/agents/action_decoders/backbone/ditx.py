"""Single-time DiT-X backbone for standard conditional flow matching."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import RmsNorm

from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.position_encodings import TimestepMLP
from dexmani_policy.utils.validation import positive_int

from .consistency_ditx import DiTXBlock, FinalLayer, WEIGHT_INIT_STD


class DiTX(nn.Module):
    """Predict action velocity from noisy actions and observation tokens.

    Each of the eight default blocks reuses the existing action self-attention,
    observation cross-attention and feed-forward implementation. Only the
    current flow time conditions AdaLN; there is no consistency target time.

    Context is ``[B, n_obs_steps * tokens_per_frame, obs_token_dim]`` in
    frame-major order. Tokens in one frame share temporal position embeddings;
    their spatial and modality identities belong to the observation encoder.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_steps: int,
        obs_token_dim: int,
        timestep_embed_dim: int = 128,
        n_layers: int = 8,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (
            ("horizon", horizon),
            ("action_dim", action_dim),
            ("n_obs_steps", n_obs_steps),
            ("obs_token_dim", obs_token_dim),
            ("timestep_embed_dim", timestep_embed_dim),
            ("n_layers", n_layers),
        ):
            positive_int(value, name)
        if timestep_embed_dim % 2:
            raise ValueError("timestep_embed_dim must be even")

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.obs_token_dim = obs_token_dim
        self.hidden_dim = hidden_dim

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.context_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        self.timestep_embedder = TimestepMLP(timestep_embed_dim, hidden_dim)
        self.ditx_blocks = nn.ModuleList(
            DiTXBlock(
                hidden_size=hidden_dim,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                p_drop_attn=p_drop_attn,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            )
            for _ in range(n_layers)
        )
        self.final_layer = FinalLayer(hidden_dim, action_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)
        for block in self.ditx_blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.normal_(self.input_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.input_pos_embed, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.context_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.context_frame_pos_embed)
        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.final_layer.ffn_final.fc2.weight)
        nn.init.zeros_(self.final_layer.ffn_final.fc2.bias)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["input_pos_embed", "context_frame_pos_embed"],
            extra_blacklist=(RmsNorm,),
        )

    def _embed_context(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] != self.obs_token_dim:
            raise ValueError("context must have shape [B, L, obs_token_dim]")
        if context.shape[1] == 0 or context.shape[1] % self.n_obs_steps:
            raise ValueError("context token count must be nonzero and divisible by n_obs_steps")
        context_c = self.context_embedder(context)
        frame_pe = self.context_frame_pos_embed.repeat_interleave(
            context.shape[1] // self.n_obs_steps, dim=1
        )
        return context_c + frame_pe.to(dtype=context_c.dtype)

    def forward(self, x: torch.Tensor, timestep, context: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("x must have shape [B, horizon, action_dim]")
        context_c = self._embed_context(context)
        if context.shape[0] != x.shape[0]:
            raise ValueError("actions and context must have the same batch size")
        timestep = torch.as_tensor(timestep, dtype=torch.float32, device=x.device)
        if timestep.ndim > 1 or timestep.numel() not in (1, x.shape[0]):
            raise ValueError("timestep must be scalar or have shape [B]")
        time_c = self.timestep_embedder(timestep.reshape(-1).expand(x.shape[0]))

        x = self.input_embedder(x)
        x = x + self.input_pos_embed.to(dtype=x.dtype)
        for block in self.ditx_blocks:
            x = block(x, time_c, context_c)
        return self.final_layer(x)
