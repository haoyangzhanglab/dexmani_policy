"""Interaction-conditioned action expert and future latent readout."""

import torch
from torch import nn

from .ditx import CrossAttention, DiTX
from .modules import WEIGHT_INIT_STD, initialize_linear


class FutureInteractionHead(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.action_projection = nn.Linear(hidden_dim, latent_dim)
        self.queries = nn.Parameter(torch.empty(1, 6, latent_dim))
        self.attention = CrossAttention(latent_dim, num_heads=4)
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim), nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.apply(initialize_linear)
        nn.init.normal_(self.queries, std=WEIGHT_INIT_STD)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        memory = self.action_projection(hidden)
        queries = self.queries.expand(hidden.shape[0], -1, -1)  # Wrist-first readout queries.
        return self.mlp(self.norm(self.attention(queries, memory)))


class InteractionDiTX(DiTX):
    """DAI-Flow action expert with unified scene/interaction memory."""

    def __init__(
        self, *, n_obs_steps: int, obs_token_dim: int, num_scene_tokens: int,
        hidden_dim: int = 768, **kwargs,
    ) -> None:
        super().__init__(
            n_obs_steps=n_obs_steps, obs_token_dim=obs_token_dim, hidden_dim=hidden_dim, **kwargs,
        )
        self.condition_counts = (n_obs_steps * num_scene_tokens, 1, 6)
        self.action_start_index = n_obs_steps - 1
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.interaction_projection = nn.Linear(obs_token_dim, hidden_dim)
        self.interaction_norm = nn.LayerNorm(hidden_dim)
        self.role_embeddings = nn.Parameter(torch.empty(1, 6, hidden_dim))
        self.state_mlp = nn.Sequential(
            nn.Linear(obs_token_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.interaction_projection.apply(initialize_linear)
        self.state_mlp.apply(initialize_linear)
        nn.init.normal_(self.role_embeddings, std=WEIGHT_INIT_STD)
        self.future_head = FutureInteractionHead(hidden_dim, obs_token_dim)

    def _encode_condition(self, context):
        scene, state, interaction = context.split(self.condition_counts, dim=1)
        scene = self.scene_norm(self.embed_context(scene))
        interaction = self.interaction_projection(interaction)
        interaction = self.interaction_norm(interaction + self.role_embeddings.to(interaction.dtype))
        return torch.cat((scene, interaction), dim=1), self.state_mlp(state[:, 0])

    def forward(
        self, x: torch.Tensor, timestep, context: torch.Tensor, *,
        context_kv=None, modulations=None, return_future: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self._forward_hidden(
            x, timestep, context, context_kv=context_kv, modulations=modulations,
        )
        velocity = self.final_layer(hidden)
        if return_future:
            future_hidden = hidden[:, self.action_start_index:]
            return velocity, self.future_head(future_hidden)
        return velocity

    def sampling_modulations(self, timesteps, context_kv, chunk_size):
        state, _ = context_kv
        # Bound the per-step modulation tensor when the caller requests a longer Euler grid.
        for chunk in timesteps.split(chunk_size):
            time = self.timestep_embedder(chunk).unsqueeze(1)
            yield from self.adaln_modulation(time + state).unbind(0)
