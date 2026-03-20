import math
import torch
import torch.nn as nn
from torch.jit import Final
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, use_fused_attn

from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay

#################################################################################
#                               自注意力和FILM                                    #
#################################################################################

def modulate(x, shift, scale):
    # FILM调制，把 γ 设为 (1+scale)，把 β 设为 shift
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):

    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()  # fused_attn会使训练更快，并且更省显存

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)    # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0) # 沿着第0维将qkv张量拆分成q,k,v
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            if attn_mask is not None:
                attn_scores += attn_mask    # 注意力掩码，允许位置加 0，被屏蔽位置加一个“足够大的负数”
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
            x = torch.matmul(attn_weights, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


#################################################################################
#                       将扩散过程的时间步编码为向量表示                             #
#################################################################################

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]    # t的shape为 (B,), freqs的shape为 (half,), 相乘后shape为 (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # shape为 (B, 2*half)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # dim为奇数时补齐维度, embedding的shape为 (B, dim)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    


#################################################################################
#                                 DiT 核心模块                                   #
#################################################################################

class DiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True, **block_kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # Pre-Norm elementwise_affine=False表示关闭LayerNorm的可学习仿射参数
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)   # mlp_ratio代表FFN中间层的维度扩展比例
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        # pre-norm -> AdaLN调制 -> 自注意力机制/MLP -> 门控+残差连接
        # 门控机制提升模型的表达能力，并且有助于稳定训练
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask) # norm, scale&shift, attn, scale,
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, output_dim):
        super().__init__()

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

#################################################################################
#                              DiT from ScaleDP                                 #
#################################################################################
class DiT_Diffusion(nn.Module):

    def __init__(
            self,
            horizon:int, 
            action_dim:int,
            cond_dim:int,
            n_emb:int=512,
            num_heads:int=8,
            n_layers:int=12,
            mlp_ratio:float=4.0,
            qkv_bias:bool=True,
    ):
        super().__init__()

        self.horizon = horizon
        self.cond_dim = cond_dim
        self.action_dim = action_dim

        self.x_embedder = nn.Linear(action_dim, n_emb)
        self.time_embedder = TimestepEmbedder(n_emb)
        self.cond_embedder = nn.Linear(cond_dim, n_emb)
        # nn.Emedding是一个离散查表器（离散索引->查表向量）
        # nn.Parameter是一个可学习的连续张量，在这里表示位置偏置矩阵，它会被广播加到输入的嵌入向量上，提供位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, n_emb))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=n_emb, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias) for _ in range(n_layers)
        ])
        self.final_layer = FinalLayer(hidden_size=n_emb, output_dim=action_dim)

        self.initialize_weights()
    

    def initialize_weights(self):
        # 初始化transfromer层
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        w = self.x_embedder.weight.data         
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.cond_embedder.weight.data, mean=0.0, std=0.02)
        nn.init.constant_(self.cond_embedder.bias, 0)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["pos_embed"],
        )

    def forward(self, x, timestep, context):

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(x.device)
        t = timestep.expand(x.shape[0])

        x = self.x_embedder(x) + self.pos_embed.to(device=x.device, dtype=x.dtype)
        t_emb = self.time_embedder(t)

        context = context.reshape(context.shape[0], -1)
        cond_emb = self.cond_embedder(context)

        c = t_emb + cond_emb

        for block in self.blocks:
            x = block(x, c, attn_mask=None)
        x = self.final_layer(x, c)

        return x



class DiT_FlowMatch(nn.Module):

    def __init__(
            self,
            horizon:int, 
            action_dim:int,
            cond_dim:int,
            n_emb:int=512,
            num_heads:int=8,
            n_layers:int=12,
            mlp_ratio:float=4.0,
            qkv_bias:bool=True,
    ):
        super().__init__()

        self.horizon = horizon
        self.cond_dim = cond_dim
        self.action_dim = action_dim

        self.x_embedder = nn.Linear(action_dim, n_emb)
        self.cond_embedder = nn.Linear(cond_dim, n_emb)
        self.time_embedder = TimestepEmbedder(n_emb)
        self.target_t_embedder = TimestepEmbedder(n_emb)
        self.timestep_and_target_t_fusion = nn.Linear(2*n_emb, n_emb)
        # nn.Emedding是一个离散查表器（离散索引->查表向量）
        # nn.Parameter是一个可学习的连续张量，在这里表示位置偏置矩阵，它会被广播加到输入的嵌入向量上，提供位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, n_emb))


        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=n_emb, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias) for _ in range(n_layers)
        ])
        self.final_layer = FinalLayer(hidden_size=n_emb, output_dim=action_dim)

        self.initialize_weights()
    

    def initialize_weights(self):
        # 初始化transfromer层
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        w = self.x_embedder.weight.data         
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.cond_embedder.weight.data, mean=0.0, std=0.02)
        nn.init.constant_(self.cond_embedder.bias, 0)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.target_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.target_t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.timestep_and_target_t_fusion.weight, std=0.02)
        nn.init.constant_(self.timestep_and_target_t_fusion.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        return get_optim_group_with_no_decay(
            self,
            weight_decay=weight_decay,
            no_decay_names=["pos_embed"],
        )

    def forward(self, x, timestep, target_t, context):
        x = self.x_embedder(x) + self.pos_embed.to(device=x.device, dtype=x.dtype)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])
        t_emb = self.time_embedder(timestep)

        if not torch.is_tensor(target_t):
            target_t = torch.tensor([target_t], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(target_t) and len(target_t.shape) == 0:
            target_t = target_t[None].to(x.device)
        target_t = target_t.expand(x.shape[0])
        target_t_emb = self.target_t_embedder(target_t)

        time_c = torch.cat([t_emb, target_t_emb], dim=-1)
        time_c = self.timestep_and_target_t_fusion(time_c)        

        context = context.reshape(context.shape[0], -1)
        cond = self.cond_embedder(context)

        c = time_c + cond

        for block in self.blocks:
            x = block(x, c, attn_mask=None)
        x = self.final_layer(x, c)

        return x
