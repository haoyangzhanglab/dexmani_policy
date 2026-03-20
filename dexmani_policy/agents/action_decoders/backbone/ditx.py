import math
import torch
import torch.nn as nn
from torch.jit import Final
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Mlp, use_fused_attn, RmsNorm
from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay

#################################################################################
#                            正余弦位置编码                        
#################################################################################
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

#################################################################################
#                            FILM、AdaLN-Zero、交叉注意力                          
#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNZero(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()

        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_linear = nn.Linear(cond_dim, dim * 2)
        self.cond_modulation = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            self.cond_linear
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.constant_(self.cond_linear.bias[:self.dim], 1.)
        nn.init.zeros_(self.cond_linear.bias[self.dim:])
    
    def forward(self, x, cond):
        x = self.norm(x)
        gamma, beta = self.cond_modulation(cond).chunk(2, dim=-1)
        x = x * gamma + beta
        return x
    

class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,       # 注意力权重的dropout
            proj_drop: float = 0,       # 输出投影的dropout
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, c, mask=None):
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


#################################################################################
#                           Ditx块和Final Layer                          
#################################################################################
class DiTXBlock(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            num_heads, 
            mlp_ratio=4.0, 
            p_drop_attn=0., 
            qkv_bias=False, 
            qk_norm=False, 
            **block_kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size

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
            **block_kwargs
        )
       
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh") 
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For self-attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For cross-attention
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For MLP

        # AdaLN modulation
        modulation_size = 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_size, bias=True)
        )
        
    def forward(self, x, time_c, context_c, attn_mask=None):
        """
        x: 动作token, 形状为(B, horizon, hidden_size)
        time_c: 时间步条件，形状为(B, hidden_size), 用于生成AdaLN-Zero的调制参数
        context_c: 多模态输入token, 形状为(B, num, hidden_size)，用于交叉注意力
        attn_mask: 动作token自注意力的掩码, 形状为(B, horizon, horizon)
        自适应层归一化是在每个子层(自注意力、交叉注意力、MLP)之前进行的, context_c的层归一化可以在交叉注意力中实现, 也可以在Ditx Transformer中实现
        """

        modulation = self.adaLN_modulation(time_c)

        chunks = modulation.chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # Self-Attention with adaLN-zero conditioning
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, _ = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        # Cross-Attention with adaLN conditioning
        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)  
        cross_attn_output = self.cross_attn(normed_x_cross, context_c, mask=None)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output
       

        # MLP with adaLN conditioning
        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output  

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()

        # 将DitX的输出维度映射到动作空间维度
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, 
            drop=0,
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


#################################################################################
#                           Ditx Transformer                          
#################################################################################
class DiTX_FlowMatch(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_step: int,
        obs_seq_len: int,
        obs_feat_dim: int,
        timestep_embed_dim: int = 128,
        target_t_embed_dim: int = 128,
        n_layers: int = 12,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        pre_norm_modality: bool = False,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.pre_norm_modality = pre_norm_modality

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        self.context_embedder = nn.Linear(obs_feat_dim, hidden_dim)
        self.context_pos_embed = nn.Parameter(torch.zeros(1, obs_seq_len * n_obs_step, hidden_dim))
        if self.pre_norm_modality:
            self.context_norm = AdaLNZero(dim=hidden_dim, cond_dim=hidden_dim)
        
        self.timestep_embedder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, hidden_dim),
        )
        self.target_t_embedder = nn.Sequential(
            SinusoidalPosEmb(target_t_embed_dim),
            nn.Linear(target_t_embed_dim, target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(target_t_embed_dim * 4, hidden_dim),
        )

        self.timestep_and_target_t_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.ditx_blocks = nn.ModuleList([
            DiTXBlock(
                hidden_size=hidden_dim,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                p_drop_attn=p_drop_attn,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            ) for _ in range(n_layers)
        ])
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

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init) 

        for block in self.ditx_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.input_embedder.weight, std=0.02)
        nn.init.constant_(self.input_embedder.bias, 0) if self.input_embedder.bias is not None else None
        nn.init.normal_(self.input_pos_embed, std=0.02)

        nn.init.normal_(self.context_embedder.weight, std=0.02)
        nn.init.constant_(self.context_embedder.bias, 0) if self.context_embedder.bias is not None else None
        nn.init.normal_(self.context_pos_embed, std=0.02)

        if self.pre_norm_modality:
            nn.init.zeros_(self.context_norm.cond_linear.weight)
            nn.init.constant_(self.context_norm.cond_linear.bias[:self.hidden_dim], 1.) 
            nn.init.zeros_(self.context_norm.cond_linear.bias[self.hidden_dim:])

        for layer in self.timestep_embedder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for layer in self.target_t_embedder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.timestep_and_target_t_fusion.weight, std=0.02)
        nn.init.constant_(self.timestep_and_target_t_fusion.bias, 0)

        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)


    def get_optim_groups(self, weight_decay: float = 1e-3):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["input_pos_embed", "context_pos_embed"],
            extra_blacklist=(RmsNorm),
        )


    def forward(self, x, timestep, target_t, context):
        x = self.input_embedder(x) + self.input_pos_embed

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

        context_c = self.context_embedder(context) + self.context_pos_embed
        if self.pre_norm_modality:
            context_c = self.context_norm(context_c, time_c)
        
        for block in self.ditx_blocks:
            x = block(x, time_c, context_c)
        
        x = self.final_layer(x)

        x = x[:, -self.horizon:]
        return x



class DiTX_Diffusion(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_obs_step: int,
        obs_seq_len: int,
        obs_feat_dim: int,
        timestep_embed_dim: int = 128,
        n_layers: int = 12,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        pre_norm_modality: bool = False,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.pre_norm_modality = pre_norm_modality

        self.input_embedder = nn.Linear(action_dim, hidden_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        self.context_embedder = nn.Linear(obs_feat_dim, hidden_dim)
        self.context_pos_embed = nn.Parameter(torch.zeros(1, obs_seq_len * n_obs_step, hidden_dim))
        if self.pre_norm_modality:
            self.context_norm = AdaLNZero(dim=hidden_dim, cond_dim=hidden_dim)
        
        self.timestep_embedder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, hidden_dim),
        )

        self.ditx_blocks = nn.ModuleList([
            DiTXBlock(
                hidden_size=hidden_dim,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                p_drop_attn=p_drop_attn,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            ) for _ in range(n_layers)
        ])
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

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init) 

        for block in self.ditx_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.input_embedder.weight, std=0.02)
        nn.init.constant_(self.input_embedder.bias, 0) if self.input_embedder.bias is not None else None
        nn.init.normal_(self.input_pos_embed, std=0.02)

        nn.init.normal_(self.context_embedder.weight, std=0.02)
        nn.init.constant_(self.context_embedder.bias, 0) if self.context_embedder.bias is not None else None
        nn.init.normal_(self.context_pos_embed, std=0.02)

        if self.pre_norm_modality:
            nn.init.zeros_(self.context_norm.cond_linear.weight)
            nn.init.constant_(self.context_norm.cond_linear.bias[:self.hidden_dim], 1.) 
            nn.init.zeros_(self.context_norm.cond_linear.bias[self.hidden_dim:])


        for layer in self.timestep_embedder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
    

    def get_optim_groups(self, weight_decay: float = 1e-3):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["input_pos_embed", "context_pos_embed"],
            extra_blacklist=(RmsNorm),
        )


    def forward(self, x, timestep, context):
        x = self.input_embedder(x) + self.input_pos_embed

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])
        timestep_embed = self.timestep_embedder(timestep)

        time_c = timestep_embed

        context_c = self.context_embedder(context) + self.context_pos_embed
        if self.pre_norm_modality:
            context_c = self.context_norm(context_c, time_c)
        
        for block in self.ditx_blocks:
            x = block(x, time_c, context_c)
        
        x = self.final_layer(x)

        x = x[:, -self.horizon:]
        return x