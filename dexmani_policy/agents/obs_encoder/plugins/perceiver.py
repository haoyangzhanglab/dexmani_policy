import torch
import torch.nn as nn


def check_attention_mask_shape(mask, batch_size, query_len, key_len):
    expected_shape = (batch_size, query_len, key_len)
    if mask.shape != expected_shape:
        raise ValueError(f"Expected attention mask shape {expected_shape}, got {tuple(mask.shape)}")


def check_tokens_shape(tokens):
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape [B, S, D], got {tuple(tokens.shape)}")


def check_obs_emb_shape(obs_emb, obs_horizon, num_modalities, obs_dim):
    if obs_emb.ndim != 5:
        raise ValueError(f"Expected obs_emb with shape [B, N, T, L, D], got {tuple(obs_emb.shape)}")

    batch_size, input_modalities, time_steps, num_local_tokens, input_dim = obs_emb.shape

    if time_steps != obs_horizon:
        raise ValueError(f"Expected T={obs_horizon}, got T={time_steps}")
    if input_modalities != num_modalities:
        raise ValueError(f"Expected N={num_modalities}, got N={input_modalities}")
    if input_dim != obs_dim:
        raise ValueError(f"Expected obs_dim={obs_dim}, got D={input_dim}")

    return batch_size, input_modalities, time_steps, num_local_tokens, input_dim


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=None, dropout=0.0):
        super().__init__()
        context_dim = dim if context_dim is None else context_dim
        dim_head = max(1, dim // heads) if dim_head is None else dim_head
        inner_dim = heads * dim_head

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context

        batch_size, query_len, _ = x.shape
        _, key_len, _ = context.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(batch_size, query_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, key_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, key_len, self.heads, self.dim_head).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            check_attention_mask_shape(mask, batch_size, query_len, key_len)
            attn_mask = mask[:, None]
            min_value = -torch.finfo(attn_logits.dtype).max
            attn_logits = attn_logits.masked_fill(~attn_mask, min_value)

        attn = attn_logits.softmax(dim=-1)

        if mask is not None:
            attn = attn * attn_mask.to(attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, query_len, self.heads * self.dim_head)

        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=None, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm_latent = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.cross_attn = Attention(
            dim=dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, latents, tokens, mask=None):
        latents = latents + self.cross_attn(
            self.norm_latent(latents),
            context=self.norm_context(tokens),
            mask=mask,
        )
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm_attn(x), mask=mask)
        x = x + self.ff(self.norm_ff(x))
        return x


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_latents,
        depth,
        cross_heads,
        latent_heads,
        cross_dim_head=None,
        latent_dim_head=None,
        ff_mult=4,
        dropout=0.0,
        self_attn_causal=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.self_attn_causal = self_attn_causal

        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        self.cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            context_dim=input_dim,
            heads=cross_heads,
            dim_head=cross_dim_head,
            ff_mult=ff_mult,
            dropout=dropout,
        )

        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=latent_dim,
                heads=latent_heads,
                dim_head=latent_dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

    def build_causal_mask(self, batch_size, device):
        mask = torch.ones(self.num_latents, self.num_latents, dtype=torch.bool, device=device).tril()
        return mask.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, tokens, cross_mask=None, self_mask=None):
        check_tokens_shape(tokens)

        batch_size = tokens.shape[0]
        latents = self.latents.expand(batch_size, -1, -1)
        latents = self.cross_attn(latents, tokens, mask=cross_mask)

        if self.self_attn_causal:
            causal_mask = self.build_causal_mask(batch_size, tokens.device)
            self_mask = causal_mask if self_mask is None else (self_mask & causal_mask)

        for block in self.self_attn_blocks:
            latents = block(latents, mask=self_mask)

        return latents


class CausalObservationPerceiver(nn.Module):
    def __init__(
        self,
        obs_dim,
        latent_dim,
        obs_horizon,
        num_modalities,
        depth=2,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=None,
        latent_dim_head=None,
        ff_mult=4,
        dropout=0.0,
        use_time_embedding=True,
        use_modality_embedding=True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.obs_horizon = obs_horizon
        self.num_modalities = num_modalities

        if use_time_embedding:
            self.time_embed = nn.Parameter(torch.randn(obs_horizon, obs_dim))
        else:
            self.register_parameter("time_embed", None)

        if use_modality_embedding:
            self.modality_embed = nn.Parameter(torch.randn(num_modalities, obs_dim))
        else:
            self.register_parameter("modality_embed", None)

        self.input_norm = nn.LayerNorm(obs_dim)

        self.encoder = PerceiverEncoder(
            input_dim=obs_dim,
            latent_dim=latent_dim,
            num_latents=obs_horizon,
            depth=depth,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            ff_mult=ff_mult,
            dropout=dropout,
            self_attn_causal=True,
        )

    def build_causal_masks(self, batch_size, time_steps, num_modalities, num_local_tokens, device):
        latent_keep = torch.ones(time_steps, time_steps, dtype=torch.bool, device=device).tril()

        cross_keep = latent_keep[:, :, None, None]
        cross_keep = cross_keep.expand(-1, -1, num_modalities, num_local_tokens)
        cross_keep = cross_keep.reshape(time_steps, time_steps * num_modalities * num_local_tokens)

        latent_keep = latent_keep.unsqueeze(0).expand(batch_size, -1, -1)
        cross_keep = cross_keep.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_keep, cross_keep

    def forward(self, obs_emb):
        batch_size, num_modalities, time_steps, num_local_tokens, obs_dim = check_obs_emb_shape(
            obs_emb,
            self.obs_horizon,
            self.num_modalities,
            self.obs_dim,
        )

        if self.modality_embed is not None:
            obs_emb = obs_emb + self.modality_embed[None, :, None, None, :]

        if self.time_embed is not None:
            obs_emb = obs_emb + self.time_embed[None, None, :, None, :]

        obs_tokens = obs_emb.permute(0, 2, 1, 3, 4).contiguous()
        obs_tokens = obs_tokens.view(batch_size, time_steps * num_modalities * num_local_tokens, obs_dim)
        obs_tokens = self.input_norm(obs_tokens)

        latent_keep_mask, cross_keep_mask = self.build_causal_masks(
            batch_size=batch_size,
            time_steps=time_steps,
            num_modalities=num_modalities,
            num_local_tokens=num_local_tokens,
            device=obs_emb.device,
        )

        return self.encoder(
            obs_tokens,
            cross_mask=cross_keep_mask,
            self_mask=latent_keep_mask,
        )


def example():
    torch.manual_seed(0)

    print("=== Standard PerceiverEncoder ===")
    encoder = PerceiverEncoder(
        input_dim=64,
        latent_dim=128,
        num_latents=8,
        depth=2,
        cross_heads=4,
        latent_heads=4,
    )

    tokens = torch.randn(2, 50, 64)
    latents = encoder(tokens)

    print("tokens  :", tuple(tokens.shape))
    print("latents :", tuple(latents.shape))

    print("\n=== CausalObservationPerceiver ===")
    causal_encoder = CausalObservationPerceiver(
        obs_dim=64,
        latent_dim=128,
        obs_horizon=4,
        num_modalities=3,
        depth=2,
        cross_heads=4,
        latent_heads=4,
    )

    obs_emb = torch.randn(2, 3, 4, 6, 64)
    sequence = causal_encoder(obs_emb)

    print("obs_emb  :", tuple(obs_emb.shape))
    print("sequence :", tuple(sequence.shape))
    print("meaning  : [B, T, D_latent], one latent per timestep")


if __name__ == "__main__":
    example()