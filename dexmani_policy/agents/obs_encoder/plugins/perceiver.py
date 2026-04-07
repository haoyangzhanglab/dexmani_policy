import torch
import torch.nn as nn


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
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context

        batch_size, query_len, _ = x.shape
        key_len = context.shape[1]

        q = self.to_q(x).view(batch_size, query_len, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(context).view(batch_size, key_len, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(batch_size, key_len, self.heads, self.dim_head).transpose(1, 2)

        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            if mask.shape != (batch_size, query_len, key_len):
                raise ValueError(
                    f"Expected attention mask shape {(batch_size, query_len, key_len)}, got {tuple(mask.shape)}"
                )
            attn_mask = mask[:, None]
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)

        if mask is not None:
            attn = attn * attn_mask.to(attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        attn = self.attn_dropout(attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.heads * self.dim_head)
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=None, ff_mult=4, dropout=0.0):
        super().__init__()
        self.latent_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attn = Attention(
            dim=dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, latents, tokens, mask=None):
        latents = latents + self.cross_attn(
            self.latent_norm(latents),
            context=self.context_norm(tokens),
            mask=mask,
        )
        latents = latents + self.ff(self.ff_norm(latents))
        return latents


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class PerceiverEncoder(nn.Module):
    """
    Perceiver 的核心想法是:
        用一小组可学习的 latent queries 去读取一大组输入 tokens。

    这样做的目的不是把所有 token 彼此两两做 self-attention，
    而是先把信息压到 M 个 latent 上，再只在 latent 空间里做建模。
    当输入 token 很多时，这比直接在 token 维做全连接注意力更省。

    这里实现的是一个最小的 latent encoder:
        tokens --cross-attn--> latents --self-attn--> refined latents

    输入:
        tokens: [B, S, D_in]
    输出:
        latents: [B, M, D_latent]
    """
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

    def _build_causal_mask(self, batch_size, device):
        mask = torch.ones(self.num_latents, self.num_latents, dtype=torch.bool, device=device).tril()
        return mask.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, tokens, cross_mask=None, self_mask=None):
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B, S, D], got {tuple(tokens.shape)}")

        batch_size = tokens.shape[0]
        latents = self.latents.expand(batch_size, -1, -1)
        latents = self.cross_attn(latents, tokens, mask=cross_mask)

        if self.self_attn_causal:
            causal_mask = self._build_causal_mask(batch_size, tokens.device)
            self_mask = causal_mask if self_mask is None else (self_mask & causal_mask)

        for block in self.self_attn_blocks:
            latents = block(latents, mask=self_mask)

        return latents


class CausalObservationPerceiver(nn.Module):
    """
    这是一个面向当前仓库时序观测的 Perceiver wrapper。

    上游会先把不同模态编码成 observation tokens:
        obs_emb: [B, N, T, L, D]

    这里做的事是:
        1. 给 token 加上时间和模态身份
        2. 把 [N, T, L] 排成 time-major 的 token 序列
        3. 用 T 个 latent queries 去读取这些 tokens
        4. 加上 causal mask, 让第 t 个 latent 只能看 <= t 的观测

    所以它更像一个“时序条件编码器”:
        [B, N, T, L, D] -> [B, T, D_latent]
    """
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

    def _build_causal_masks(self, batch_size, time_steps, num_modalities, num_local_tokens, device):
        latent_keep = torch.ones(time_steps, time_steps, dtype=torch.bool, device=device).tril()
        cross_keep = (
            latent_keep[:, :, None, None]
            .expand(-1, -1, num_modalities, num_local_tokens)
            .reshape(time_steps, time_steps * num_modalities * num_local_tokens)
        )
        latent_keep = latent_keep.unsqueeze(0).expand(batch_size, -1, -1)
        cross_keep = cross_keep.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_keep, cross_keep

    def forward(self, obs_emb):
        if obs_emb.ndim != 5:
            raise ValueError(f"Expected obs_emb with shape [B, N, T, L, D], got {tuple(obs_emb.shape)}")

        batch_size, num_modalities, time_steps, num_local_tokens, obs_dim = obs_emb.shape
        if time_steps != self.obs_horizon:
            raise ValueError(f"Expected T={self.obs_horizon}, got T={time_steps}")
        if num_modalities != self.num_modalities:
            raise ValueError(f"Expected N={self.num_modalities}, got N={num_modalities}")
        if obs_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got D={obs_dim}")

        if self.modality_embed is not None:
            obs_emb = obs_emb + self.modality_embed[None, :, None, None, :]
        if self.time_embed is not None:
            obs_emb = obs_emb + self.time_embed[None, None, :, None, :]

        obs_tokens = obs_emb.permute(0, 2, 1, 3, 4).contiguous()
        obs_tokens = obs_tokens.reshape(batch_size, time_steps * num_modalities * num_local_tokens, obs_dim)
        obs_tokens = self.input_norm(obs_tokens)

        latent_keep_mask, cross_keep_mask = self._build_causal_masks(
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
    print("obs_emb   :", tuple(obs_emb.shape))
    print("sequence  :", tuple(sequence.shape))
    print("meaning   : [B, T, D_latent], one latent per timestep")


if __name__ == "__main__":
    example()
