import math
import torch
import torch.nn as nn


def gather_tokens(x, idx):
    idx = idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
    return torch.gather(x, 1, idx)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h


class CrossAttn(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, q, kv, kv_mask=None):
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        h, _ = self.attn(qn, kvn, kvn, key_padding_mask=kv_mask, need_weights=False)
        return q + h


class LatentBlock(nn.Module):
    def __init__(self, dim, num_heads, self_attn=False, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttn(dim, num_heads, dropout) if self_attn else None
        self.cross_attn = CrossAttn(dim, num_heads, dropout)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, latents, tokens, token_mask=None):
        if self.self_attn is not None:
            latents = self.self_attn(latents)
        latents = self.cross_attn(latents, tokens, kv_mask=token_mask)
        latents = self.mlp(latents)
        return latents


class IdentityReducer(nn.Module):
    def forward(self, tokens, cond=None, token_mask=None):
        return tokens, token_mask, {}


class TopKReducer(nn.Module):
    def __init__(self, dim, keep_tokens=None, keep_ratio=None, cond_dim=None):
        super().__init__()
        self.keep_tokens = keep_tokens
        self.keep_ratio = keep_ratio
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.cond_proj = nn.Linear(cond_dim, dim) if cond_dim is not None else None

    def forward(self, tokens, cond=None, token_mask=None):
        b, n, _ = tokens.shape

        h = self.norm(tokens)
        if cond is not None and self.cond_proj is not None:
            h = h + self.cond_proj(cond).unsqueeze(1)

        score = self.score(h).squeeze(-1)

        if token_mask is not None:
            score = score.masked_fill(token_mask, float("-inf"))

        if self.keep_tokens is not None:
            k = min(self.keep_tokens, n)
        else:
            k = max(1, int(n * self.keep_ratio))

        keep_idx = torch.topk(score, k, dim=1).indices
        kept_tokens = gather_tokens(tokens, keep_idx)
        kept_mask = torch.zeros(b, k, dtype=torch.bool, device=tokens.device)

        aux = {
            "score": score,
            "keep_idx": keep_idx,
        }
        return kept_tokens, kept_mask, aux


class TokenCompressor(nn.Module):
    def __init__(
        self,
        token_dim,
        latent_dim=256,
        num_latents=32,
        depth=4,
        num_heads=8,
        reducer=None,
        latent_self_attn=False,
        mlp_ratio=4.0,
        out_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.token_proj = nn.Identity() if token_dim == latent_dim else nn.Linear(token_dim, latent_dim)
        self.default_queries = nn.Parameter(
            torch.randn(1, num_latents, latent_dim) / math.sqrt(latent_dim)
        )
        self.reducer = reducer if reducer is not None else IdentityReducer()
        self.blocks = nn.ModuleList([
            LatentBlock(
                dim=latent_dim,
                num_heads=num_heads,
                self_attn=latent_self_attn,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Identity() if out_dim is None or out_dim == latent_dim else nn.Linear(latent_dim, out_dim)

    def forward(
        self,
        tokens,
        token_mask=None,
        query_tokens=None,
        reducer_cond=None,
        return_aux=False,
    ):
        num_input_tokens = tokens.size(1)

        tokens = self.token_proj(tokens)
        tokens, token_mask, reducer_aux = self.reducer(
            tokens=tokens,
            cond=reducer_cond,
            token_mask=token_mask,
        )

        if query_tokens is None:
            query_tokens = self.default_queries.expand(tokens.size(0), -1, -1)

        latents = query_tokens
        for blk in self.blocks:
            latents = blk(latents, tokens, token_mask=token_mask)

        latents = self.out_proj(self.out_norm(latents))

        if not return_aux:
            return latents

        aux = {
            "num_input_tokens": num_input_tokens,
            "num_used_tokens": tokens.size(1),
            "reducer": reducer_aux,
        }
        return latents, aux


class IdentityModulator(nn.Module):
    def forward(self, tokens, queries, cond):
        return tokens, queries, None, {}


class QueryFiLM(nn.Module):
    def __init__(self, dim, cond_dim, pass_cond_to_reducer=True):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, dim * 2)
        self.pass_cond_to_reducer = pass_cond_to_reducer

    def forward(self, tokens, queries, cond):
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        queries = queries * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        reducer_cond = cond if self.pass_cond_to_reducer else None
        aux = {
            "gamma": gamma,
            "beta": beta,
        }
        return tokens, queries, reducer_cond, aux


class ModulatedTokenCompressor(nn.Module):
    def __init__(self, compressor, modulator=None):
        super().__init__()
        self.compressor = compressor
        self.modulator = modulator if modulator is not None else IdentityModulator()

    def forward(
        self,
        tokens,
        cond,
        token_mask=None,
        query_tokens=None,
        return_aux=False,
    ):
        if query_tokens is None:
            query_tokens = self.compressor.default_queries.expand(tokens.size(0), -1, -1)

        tokens, query_tokens, reducer_cond, mod_aux = self.modulator(
            tokens=tokens,
            queries=query_tokens,
            cond=cond,
        )

        out = self.compressor(
            tokens=tokens,
            token_mask=token_mask,
            query_tokens=query_tokens,
            reducer_cond=reducer_cond,
            return_aux=return_aux,
        )

        if not return_aux:
            return out

        latents, comp_aux = out
        aux = {
            "compressor": comp_aux,
            "modulator": mod_aux,
        }
        return latents, aux


if __name__ == "__main__":
    torch.manual_seed(0)

    b = 2
    n = 192
    token_dim = 384
    latent_dim = 256
    action_dim = 24

    phase_dim = 16
    goal_dim = 16
    hand_dim = 16
    cond_dim = phase_dim + goal_dim + hand_dim

    aligned_tokens = torch.randn(b, n, token_dim)

    token_mask = torch.zeros(b, n, dtype=torch.bool)
    token_mask[:, 180:] = True

    phase_embed = torch.randn(b, phase_dim)
    goal_embed = torch.randn(b, goal_dim)
    hand_state = torch.randn(b, hand_dim)
    task_cond = torch.cat([phase_embed, goal_embed, hand_state], dim=-1)

    compressor = TokenCompressor(
        token_dim=token_dim,
        latent_dim=latent_dim,
        num_latents=32,
        depth=4,
        num_heads=8,
        reducer=TopKReducer(
            dim=latent_dim,
            keep_tokens=96,
            cond_dim=cond_dim,
        ),
        latent_self_attn=True,
        out_dim=latent_dim,
    )

    policy_tokens, aux = compressor(
        tokens=aligned_tokens,
        token_mask=token_mask,
        return_aux=True,
    )

    print("compressor")
    print("policy_tokens:", policy_tokens.shape)
    print("num_input_tokens:", aux["num_input_tokens"])
    print("num_used_tokens :", aux["num_used_tokens"])

    wrapped_compressor = ModulatedTokenCompressor(
        compressor=compressor,
        modulator=QueryFiLM(
            dim=latent_dim,
            cond_dim=cond_dim,
            pass_cond_to_reducer=True,
        ),
    )

    policy_tokens, aux = wrapped_compressor(
        tokens=aligned_tokens,
        cond=task_cond,
        token_mask=token_mask,
        return_aux=True,
    )

    print("wrapped compressor")
    print("policy_tokens:", policy_tokens.shape)
    print("num_input_tokens:", aux["compressor"]["num_input_tokens"])
    print("num_used_tokens :", aux["compressor"]["num_used_tokens"])

    phase_query_bank = torch.randn(b, 24, latent_dim)

    policy_tokens = wrapped_compressor(
        tokens=aligned_tokens,
        cond=task_cond,
        token_mask=token_mask,
        query_tokens=phase_query_bank,
        return_aux=False,
    )

    print("wrapped compressor + custom query bank")
    print("policy_tokens:", policy_tokens.shape)

    action_head = nn.Sequential(
        nn.LayerNorm(latent_dim),
        nn.Linear(latent_dim, latent_dim),
        nn.GELU(),
        nn.Linear(latent_dim, action_dim),
    )

    pooled_tokens = policy_tokens.mean(dim=1)
    action = action_head(pooled_tokens)

    print("action:", action.shape)