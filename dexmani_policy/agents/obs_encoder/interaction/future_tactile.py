"""Visuo-tactile interaction, future-force diffusion and force-guided fusion."""

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.modules import (
    SinusoidalTimestepEmbedding,
    batch_timesteps,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion


class ForcePredictionTransformer(nn.Module):
    def __init__(
        self,
        horizon: int,
        cond_dim: int,
        force_dim: int,
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_emb = nn.Linear(force_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.time_emb = SinusoidalTimestepEmbedding(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 2, n_emb))
        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.register_buffer(
            "mask", torch.full((horizon, horizon), float("-inf")).triu(1),
        )
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, force_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, ForcePredictionTransformer):
            nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, timestep, context: torch.Tensor) -> torch.Tensor:
        timesteps = batch_timesteps(timestep, x)
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        input_emb = self.input_emb(x)
        cond_embeddings = torch.cat((time_emb, self.cond_obs_emb(context)), dim=1)
        memory = self.encoder(self.drop(cond_embeddings + self.cond_pos_emb))
        x = self.drop(input_emb + self.pos_emb)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask)
        return self.head(self.ln_f(x))


class DiffusionForceHead(Diffusion):
    def __init__(
        self,
        model: nn.Module,
        num_training_steps: int = 100,
        num_inference_steps: int = 20,
    ) -> None:
        super().__init__(
            model,
            num_training_steps,
            num_inference_steps,
            prediction_type="sample",
            clip_sample=False,
        )

    def compute_loss_and_prediction(
        self, condition: torch.Tensor, force: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        force = force.float()
        noise = torch.randn_like(force)
        timestep = torch.randint(
            self.noise_scheduler.config.num_train_timesteps,
            (force.shape[0],),
            device=force.device,
        )
        prediction = self.model(
            x=self.noise_scheduler.add_noise(force, noise, timestep),
            timestep=timestep,
            context=condition,
        )
        prediction = prediction.float()
        loss = ((prediction - force).square() * valid[..., None]).sum() / (force.shape[-1] * valid.float().sum())
        return loss, prediction


class ForceGuidedAttentionFusion(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        tactile_dim: int,
        guide_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)
        self.force_proj = nn.Linear(guide_dim, hidden_dim)
        self.cross_attn = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, visual_feat: torch.Tensor, tactile_feat: torch.Tensor, force_guide: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.visual_proj(visual_feat)
        tactile = self.tactile_proj(tactile_feat)
        query = self.force_proj(force_guide).unsqueeze(1)
        memory = torch.stack([visual, tactile], dim=1)
        attended_feat = self.cross_attn(tgt=query, memory=memory)
        return attended_feat.squeeze(1)


class VisuoTactileInteraction(nn.Module):
    def __init__(self, visual_dim: int, tactile_dim: int, num_fingers: int) -> None:
        super().__init__()
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.tactile_norm = nn.LayerNorm(tactile_dim)
        self.tactile_attn = nn.MultiheadAttention(
            embed_dim=tactile_dim, num_heads=8, kdim=visual_dim, vdim=visual_dim,
            dropout=0.1, batch_first=True,
        )
        self.visual_attn = nn.MultiheadAttention(
            embed_dim=visual_dim, num_heads=8, kdim=tactile_dim, vdim=tactile_dim,
            dropout=0.1, batch_first=True,
        )
        self.tactile_dropout = nn.Dropout(0.1)
        self.visual_dropout = nn.Dropout(0.1)
        self.finger_embedding = nn.Parameter(torch.empty(1, num_fingers, tactile_dim))
        nn.init.normal_(self.finger_embedding, mean=0.0, std=0.02)

    def forward(
        self, visual: torch.Tensor, tactile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.visual_norm(visual)
        t = self.tactile_norm(tactile)
        e = self.finger_embedding.to(dtype=t.dtype)
        # Both directions read the same pre-update tokens; identity only enters Q/K.
        delta_t = self.tactile_attn(query=t + e, key=v, value=v, need_weights=False)[0]
        delta_v = self.visual_attn(query=v, key=t + e, value=t, need_weights=False)[0]
        return (
            visual + self.visual_dropout(delta_v),
            tactile + self.tactile_dropout(delta_t),
        )
