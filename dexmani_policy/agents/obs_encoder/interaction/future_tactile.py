import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.modules import (
    SinusoidalTimestepEmbedding,
    batch_timesteps,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion


class ForceTransformer(nn.Module):
    def __init__(
        self,
        horizon: int,
        cond_dim: int,
        force_dim: int,
        depth: int = 8,
        num_heads: int = 4,
        embed_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_emb = nn.Linear(force_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, embed_dim))
        self.time_emb = SinusoidalTimestepEmbedding(embed_dim)
        self.cond_obs_emb = nn.Linear(cond_dim, embed_dim)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.condition_mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.Mish(), nn.Linear(4 * embed_dim, embed_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.register_buffer(
            "causal_mask", torch.full((horizon, horizon), float("-inf")).triu(1),
            persistent=False,
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, force_dim)
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
        elif isinstance(module, ForceTransformer):
            nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, timestep, context: torch.Tensor) -> torch.Tensor:
        timesteps = batch_timesteps(timestep, x)
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        input_emb = self.input_emb(x)
        cond_embeddings = torch.cat((time_emb, self.cond_obs_emb(context)), dim=1)
        memory = self.condition_mlp(cond_embeddings + self.cond_pos_emb)
        x = input_emb + self.pos_emb
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.causal_mask)
        return self.head(self.ln_f(x))


class ForceDiffusion(Diffusion):
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
        error = (prediction - force).square() * valid[..., None]
        loss = error.sum() / (force.shape[-1] * valid.float().sum())
        return loss, prediction


class ForceGuidedFusion(nn.Module):
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
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, visual_feat: torch.Tensor, tactile_feat: torch.Tensor, force_guide: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.visual_proj(visual_feat)
        tactile = self.tactile_proj(tactile_feat)
        query = self.force_proj(force_guide).unsqueeze(1)
        memory = torch.stack([visual, tactile], dim=1)
        update = self.cross_attn(query, memory, memory, need_weights=False)[0]
        query = self.attn_norm(query + self.dropout(update))
        return self.ffn_norm(query + self.dropout(self.ffn(query))).squeeze(1)


class VisuoTactileAttention(nn.Module):
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
