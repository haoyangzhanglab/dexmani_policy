import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Mlp, RmsNorm

from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
from dexmani_policy.agents.position_encodings import TimestepMLP

from .attention import CrossAttention
from .dit import _approx_gelu, modulate

WEIGHT_INIT_STD = 0.02


class AdaLNZero(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()

        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_linear = nn.Linear(cond_dim, dim * 2)
        self.cond_modulation = nn.Sequential(Rearrange("b d -> b 1 d"), nn.SiLU(), self.cond_linear)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.constant_(self.cond_linear.bias[: self.dim], 1.0)
        nn.init.zeros_(self.cond_linear.bias[self.dim :])

    def forward(self, x, cond):
        x = self.norm(x)
        gamma, beta = self.cond_modulation(cond).chunk(2, dim=-1)
        x = x * gamma + beta
        return x


class DiTXBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        p_drop_attn=0.0,
        qkv_bias=False,
        qk_norm=False,
    ):
        super().__init__()

        # Official ManiFlow self-attention always has QKV bias, without QK norm.
        # qkv_bias/qk_norm configure only the custom cross-attention below.
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            dropout=p_drop_attn,
        )

        self.cross_attn = CrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=_approx_gelu, drop=0.0
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        modulation_size = 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, modulation_size, bias=True))

    def forward(self, x, time_c, context_c, attn_mask=None):
        modulation = self.adaLN_modulation(time_c)

        chunks = modulation.chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output = self.self_attn(
            normed_x, normed_x, normed_x, attn_mask=attn_mask, need_weights=False
        )[0]
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_attn_output = self.cross_attn(normed_x_cross, context_c, mask=None)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output

        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()

        self.norm_final = RmsNorm(hidden_size, eps=1e-6)

        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=_approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class ConsistencyDiTX(nn.Module):
    """DiT-X backbone for ManiFlow consistency flow matching.

    Each action token contains the full action vector for one timestep.
    Blocks apply temporal self-attention, action-to-observation cross-attention
    and an MLP, conditioned by fused timestep/target_t embeddings via AdaLN-Zero.

    Context must contain equally sized frames in frame-major order. Projected
    tokens share a learned position embedding within each observation frame;
    point indices have no positional identity. ``qkv_bias`` and ``qk_norm``
    configure only cross-attention; self-attention uses biased PyTorch MHA.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_steps: int,
        obs_token_dim: int,
        timestep_embed_dim: int = 128,
        target_t_embed_dim: int = 128,
        n_layers: int = 12,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        pre_norm_modality: bool = False,
    ):
        super().__init__()
        if n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be greater than 0")

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.pre_norm_modality = pre_norm_modality

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        self.context_embedder = nn.Linear(obs_token_dim, hidden_dim)
        self.context_frame_pos_embed = nn.Parameter(torch.zeros(1, n_obs_steps, hidden_dim))
        if self.pre_norm_modality:
            self.context_norm = AdaLNZero(dim=hidden_dim, cond_dim=hidden_dim)

        self.timestep_embedder = TimestepMLP(pos_emb_dim=timestep_embed_dim, output_dim=hidden_dim)
        self.target_t_embedder = TimestepMLP(pos_emb_dim=target_t_embed_dim, output_dim=hidden_dim)

        self.timestep_and_target_t_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.ditx_blocks = nn.ModuleList(
            [
                DiTXBlock(
                    hidden_size=hidden_dim,
                    num_heads=n_head,
                    mlp_ratio=mlp_ratio,
                    p_drop_attn=p_drop_attn,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, action_dim)

        self.initialize_weights()

    def initialize_weights(self):

        for block in self.ditx_blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)

        def init_fn(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(init_fn)

        for block in self.ditx_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.pre_norm_modality:
            self.context_norm.initialize_weights()

        nn.init.normal_(self.input_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.constant_(self.input_embedder.bias, 0) if self.input_embedder.bias is not None else None
        nn.init.normal_(self.input_pos_embed, std=WEIGHT_INIT_STD)

        nn.init.normal_(self.context_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.constant_(self.context_embedder.bias, 0) if self.context_embedder.bias is not None else None
        nn.init.zeros_(self.context_frame_pos_embed)

        for layer in self.timestep_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for layer in self.target_t_embedder.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=WEIGHT_INIT_STD)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.timestep_and_target_t_fusion.weight, std=WEIGHT_INIT_STD)
        nn.init.constant_(self.timestep_and_target_t_fusion.bias, 0)

        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["input_pos_embed", "context_frame_pos_embed"],
            extra_blacklist=(RmsNorm,),
        )

    def forward(self, x, timestep, target_t, context):
        x = self.input_embedder(x) + self.input_pos_embed.to(dtype=x.dtype)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])
        timestep_embed = self.timestep_embedder(timestep)

        if not torch.is_tensor(target_t):
            target_t = torch.tensor([target_t], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(target_t) and len(target_t.shape) == 0:
            target_t = target_t[None].to(x.device)
        target_t = target_t.expand(x.shape[0])
        target_t_embed = self.target_t_embedder(target_t)

        time_c = self.timestep_and_target_t_fusion(torch.cat([timestep_embed, target_t_embed], dim=-1))

        if context.shape[1] == 0 or context.shape[1] % self.n_obs_steps != 0:
            raise ValueError("context token count must be nonzero and divisible by n_obs_steps")
        context_c = self.context_embedder(context)
        # Flattening is frame-major: every point in a frame shares its PE.
        frame_pe = self.context_frame_pos_embed.repeat_interleave(
            context.shape[1] // self.n_obs_steps, dim=1
        )
        context_c = context_c + frame_pe.to(dtype=context_c.dtype)
        if self.pre_norm_modality:
            context_c = self.context_norm(context_c, time_c)

        for block in self.ditx_blocks:
            x = block(x, time_c, context_c)

        x = self.final_layer(x)

        x = x[:, -self.horizon :]
        return x
