import torch
from torch import nn

from .ditx import DiTX
from .modules import WEIGHT_INIT_STD, batch_timesteps


class InterActDiTX(DiTX):
    def __init__(self, *, interaction_horizon: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.future_body_embedding = nn.Parameter(torch.empty(1, 6, self.hidden_dim))
        nn.init.normal_(self.future_body_embedding, std=WEIGHT_INIT_STD)

        start = self.n_obs_steps - 1
        core = torch.zeros(self.horizon + 6, dtype=torch.bool)
        core[start:start + interaction_horizon] = True
        core[self.horizon:] = True
        self.register_buffer(
            "interaction_mask", core[:, None] & ~core[None, :], persistent=False,
        )

    def forward(
        self, x: torch.Tensor, timestep, context: torch.Tensor, *, context_kv=None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.horizon:
            raise ValueError("actions must have shape [B, horizon, action_dim]")
        if context.shape[0] != x.shape[0]:
            raise ValueError("actions and context must have the same batch size")
        context_c = self.embed_context(context) if context_kv is None else None
        timestep = batch_timesteps(timestep, x, dtype=torch.float32)
        time_c = self.timestep_embedder(timestep)

        # Rebuild dynamic states on every velocity evaluation, including cached inference.
        body = context.unflatten(1, (self.n_obs_steps, -1))[:, -1, :6]
        interaction = self.context_embedder(body)
        # Remove singleton batch axes so cat gradients match parameter strides for fused AdamW at B=1.
        interaction = interaction + self.future_body_embedding.squeeze(0).to(interaction.dtype)

        actions = self.input_embedder(x)
        actions = actions + self.input_pos_embed.squeeze(0).to(actions.dtype)
        hidden = torch.cat((actions, interaction), dim=1)
        hidden = self._run_blocks(
            hidden, time_c, context_c,
            context_kv=context_kv, attn_mask=self.interaction_mask,
        )

        features = self.final_layer.norm_final(hidden)
        velocity = self.final_layer.ffn_final(features[:, :self.horizon])
        return (velocity, features) if return_features else velocity
