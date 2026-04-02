import torch
import torch.nn as nn


class CrossAttentionOnly(nn.Module):
    """
    等价于原先:
        TransformerBlock(dim, cond_dim=obs_dim, cross_attn_only=True)

    逻辑:
        x = x + CrossAttn(LN(x), LN(cond))
        x = x + FFN(LN(x))
    """
    def __init__(self, dim, cond_dim, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.heads = heads

        self.norm_x = nn.LayerNorm(dim)
        self.norm_cond = nn.LayerNorm(cond_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
            kdim=cond_dim,
            vdim=cond_dim,
        )

        hidden_dim = dim * mlp_ratio
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def _to_mha_attn_mask(self, keep_mask: torch.Tensor) -> torch.Tensor:
        """
        原代码 cond_mask 语义:
            True  = 允许 attend
            False = 禁止 attend

        nn.MultiheadAttention 的 bool attn_mask 语义:
            True  = 不允许 attend
            False = 允许 attend

        并且 3D mask 需要形状:
            [B * num_heads, L, S]
        """
        # [B, L, S] -> [B, H, L, S] -> [B*H, L, S]
        block_mask = ~keep_mask
        block_mask = block_mask[:, None, :, :].expand(-1, self.heads, -1, -1)
        block_mask = block_mask.reshape(
            keep_mask.shape[0] * self.heads,
            keep_mask.shape[1],
            keep_mask.shape[2],
        )
        return block_mask

    def forward(self, x, cond, cond_mask=None):
        q = self.norm_x(x)
        kv = self.norm_cond(cond)

        attn_mask = None
        if cond_mask is not None:
            attn_mask = self._to_mha_attn_mask(cond_mask)

        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            attn_mask=attn_mask,
            need_weights=False,  # 更利于优化路径
        )
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        return x


class CausalObservationPerceiver(nn.Module):
    def __init__(
        self,
        dim,
        obs_dim,
        obs_horizon,
        layers=1,
        heads=8,
        mlp_ratio=4,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.obs_horizon = obs_horizon
        self.heads = heads

        self.latents = nn.Parameter(torch.randn(1, obs_horizon, dim))
        self.obs_norm = nn.LayerNorm(obs_dim)

        # cross-attention only
        self.x_attn = CrossAttentionOnly(
            dim=dim,
            cond_dim=obs_dim,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # causal self-attention stack on latents
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # 更接近常见 pre-norm block
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=layers,
        )

    def _build_keep_masks(self, B, N, T, L, device):
        """
        返回:
            latent_keep_mask: [B, T, T]
                第 t 个 latent 只能看 <= t 的 latent

            cond_keep_mask: [B, T, N*T*L]
                第 t 个 latent 只能看 <= t 时刻的 observation tokens
        """
        # [T, T], True 表示允许
        latent_keep = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        latent_keep = latent_keep.unsqueeze(0).expand(B, -1, -1)  # [B, T, T]

        # 按时间展开到 observation token 维
        # 原始 obs 是 [B, N, T, L, D]，flatten(1, -2) 后 token 顺序与这里一致
        cond_keep = (
            latent_keep[:, :, None, :, None]      # [B, T, 1, T, 1]
            .expand(-1, -1, N, -1, L)             # [B, T, N, T, L]
            .reshape(B, T, N * T * L)             # [B, T, N*T*L]
        )

        return latent_keep, cond_keep

    def _to_encoder_mask(self, keep_mask: torch.Tensor) -> torch.Tensor:
        """
        TransformerEncoder / TransformerEncoderLayer 的 bool mask:
            True  = 不允许 attend
            False = 允许 attend

        3D mask 需要:
            [B * num_heads, T, T]
        """
        block_mask = ~keep_mask
        block_mask = block_mask[:, None, :, :].expand(-1, self.heads, -1, -1)
        block_mask = block_mask.reshape(
            keep_mask.shape[0] * self.heads,
            keep_mask.shape[1],
            keep_mask.shape[2],
        )
        return block_mask

    def forward(self, obs_emb):
        """
        obs_emb: [B, N, T, L, obs_dim]
        return : [B, T, dim]
        """
        B, N, T, L, D_obs = obs_emb.shape
        if T != self.obs_horizon:
            raise ValueError(f"Expected T={self.obs_horizon}, got T={T}")

        # [B, N, T, L, D_obs] -> [B, N*T*L, D_obs]
        obs_tokens = obs_emb.flatten(1, -2)
        obs_tokens = self.obs_norm(obs_tokens)

        latent_keep_mask, cond_keep_mask = self._build_keep_masks(
            B=B, N=N, T=T, L=L, device=obs_emb.device
        )

        # [1, T, dim] -> [B, T, dim]
        latents = self.latents.expand(B, -1, -1)

        # 1) cross-attn from observation tokens
        latents = self.x_attn(
            latents,
            obs_tokens,
            cond_mask=cond_keep_mask,
        )

        # 2) causal self-attn among latents
        latent_attn_mask = self._to_encoder_mask(latent_keep_mask)
        latents = self.blocks(
            latents,
            mask=latent_attn_mask,
        )

        return latents