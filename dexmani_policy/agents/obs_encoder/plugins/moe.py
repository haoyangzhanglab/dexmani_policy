import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MoEAux:
    expert_token_count: torch.Tensor
    expert_activation_rate: torch.Tensor
    load_balance_loss: torch.Tensor
    entropy_loss: torch.Tensor
    aux_loss: torch.Tensor
    expert_token_count_by_group: Optional[torch.Tensor] = None
    expert_activation_rate_by_group: Optional[torch.Tensor] = None


class Expert(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        d = in_dim
        layers = []
        out_dim = in_dim if out_dim is None else out_dim

        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), activation()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_experts: int = 16,
        top_k: int = 2,
        hidden_dim: int = 256,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        temperature: float = 1.0,
        residual: bool = True,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        out_dim = in_dim if out_dim is None else out_dim
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.lambda_load = lambda_load
        self.beta_entropy = beta_entropy
        self.temperature = temperature
        self.residual = residual and (in_dim == out_dim)

        self.router = nn.Linear(in_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                Expert(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    num_layers=num_layers,
                    activation=activation,
                )
                for _ in range(num_experts)
            ]
        )

    @staticmethod
    def expand_forced_tensor(t: torch.Tensor, target_rows: int, top_k: int, name: str) -> torch.Tensor:
        """将按 observation 给的 forced routing 张量扩展到 flattened token 维度。

        允许三种情况:
        - t 已有 target_rows 行: 直接 reshape 返回
        - target_rows % t.shape[0] == 0: 按倍数 repeat_interleave（隐含所有 obs token 数相同）
        - t.shape[0] == 1: 单个 observation 的路由，expand 到全部 token
        """
        t = t.reshape(-1, top_k)
        if t.shape[0] == target_rows:
            return t
        if not (target_rows % t.shape[0] == 0 or t.shape[0] == 1):
            raise ValueError(
                f"{name} shape mismatch: {name} has {t.shape[0]} rows "
                f"but target is {target_rows} flattened tokens. "
                f"{name}[0] must be 1 or divide target_rows exactly."
            )
        if target_rows % t.shape[0] == 0:
            t = t.repeat_interleave(target_rows // t.shape[0], dim=0)
        else:
            t = t.expand(target_rows, -1)
        return t

    def forward(
        self,
        x: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weight: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        num_groups: Optional[int] = None,
    ):
        if x.ndim < 2:
            raise ValueError("x must have shape [B, C] or [..., C]")

        token_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        router_logits = self.router(x) / self.temperature
        router_probs = torch.softmax(router_logits, dim=-1)

        if topk_idx is None:
            topk_weight, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        else:
            topk_idx = self.expand_forced_tensor(
                topk_idx, x.shape[0], self.top_k, "topk_idx"
            ).to(device=x.device, dtype=torch.long)
            if topk_weight is None:
                topk_weight = torch.gather(router_probs, dim=-1, index=topk_idx)
            else:
                topk_weight = self.expand_forced_tensor(
                    topk_weight, x.shape[0], self.top_k, "topk_weight"
                ).to(device=x.device, dtype=x.dtype)

        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        y = x.new_zeros(x.shape[0], self.out_dim)

        token_idx = (
            torch.arange(x.shape[0], device=x.device)
            .unsqueeze(1)
            .expand(x.shape[0], self.top_k)
            .reshape(-1)
        )
        expert_idx = topk_idx.reshape(-1)
        expert_weight = topk_weight.reshape(-1)

        active_expert_idx = torch.unique(expert_idx)
        for i in active_expert_idx.tolist():
            expert = self.experts[i]
            mask = expert_idx == i
            token_idx_i = token_idx[mask]
            y_i = expert(x.index_select(0, token_idx_i))
            y.index_add_(0, token_idx_i, y_i * expert_weight[mask].unsqueeze(-1))

        if self.residual:
            y = x + y

        y = y.reshape(*token_shape, self.out_dim)

        if not return_aux:
            return y

        # Per-expert routing load: raw top-k assignment count and count normalized by total tokens.
        expert_token_count = torch.bincount(expert_idx, minlength=self.num_experts).to(dtype=router_probs.dtype)
        expert_activation_rate = expert_token_count / float(x.shape[0])
        f_i = expert_token_count / float(x.shape[0] * self.top_k)
        p_i = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(f_i * p_i)
        entropy_loss = -(router_probs * torch.log(router_probs.clamp_min(1e-9))).sum(dim=-1).mean()
        aux_loss = self.lambda_load * load_balance_loss + self.beta_entropy * entropy_loss

        expert_token_count_by_group = None
        expert_activation_rate_by_group = None
        if num_groups is not None and num_groups > 0 and x.shape[0] % num_groups == 0:
            tokens_per_group = x.shape[0] // num_groups
            group_idx = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            group_idx = torch.div(group_idx, tokens_per_group, rounding_mode="floor")
            group_idx = group_idx.unsqueeze(1).expand(-1, self.top_k).reshape(-1)
            flat_group_expert_idx = group_idx * self.num_experts + expert_idx
            expert_token_count_by_group = torch.bincount(
                flat_group_expert_idx,
                minlength=num_groups * self.num_experts,
            ).reshape(num_groups, self.num_experts).to(dtype=router_probs.dtype)
            expert_activation_rate_by_group = expert_token_count_by_group / float(tokens_per_group)

        aux = MoEAux(
            expert_token_count=expert_token_count,
            expert_activation_rate=expert_activation_rate,
            load_balance_loss=load_balance_loss,
            entropy_loss=entropy_loss,
            aux_loss=aux_loss,
            expert_token_count_by_group=expert_token_count_by_group,
            expert_activation_rate_by_group=expert_activation_rate_by_group,
        )
        return y, aux


class MoEConditioner(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_experts: int = 16,
        top_k: int = 2,
        hidden_dim: int = 256,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        temperature: float = 1.0,
        residual: bool = True,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "out_shape"):
            raise AttributeError("encoder must have out_shape")

        self.encoder = encoder
        self.moe = MoE(
            in_dim=encoder.out_shape,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            out_dim=encoder.out_shape if out_dim is None else out_dim,
            num_layers=num_layers,
            lambda_load=lambda_load,
            beta_entropy=beta_entropy,
            temperature=temperature,
            residual=residual,
            activation=activation,
        )

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        topk_idx: Optional[torch.Tensor] = None,
        topk_weight: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        num_groups: Optional[int] = None,
    ):
        z = self.encoder(obs)
        return self.moe(
            z,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            return_aux=return_aux,
            num_groups=num_groups,
        )

    @property
    def out_shape(self) -> int:
        return self.moe.out_dim


# example
if __name__ == "__main__":
    class DummyEncoder(nn.Module):
        def __init__(self, obs_dim: int, feat_dim: int) -> None:
            super().__init__()
            self.out_shape = feat_dim
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, feat_dim),
            )

        def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
            return self.net(obs["state"])

    B = 8
    obs_dim = 24
    feat_dim = 64

    encoder = DummyEncoder(obs_dim, feat_dim)
    conditioner = MoEConditioner(encoder, num_experts=8, top_k=2, hidden_dim=128)

    obs = {"state": torch.randn(B, obs_dim)}

    z = conditioner(obs)
    print("z:", z.shape)

    z, aux = conditioner(obs, return_aux=True)
    print("z:", z.shape)
    print("expert_token_count:", aux.expert_token_count)
    print("expert_activation_rate:", aux.expert_activation_rate)
    print("aux_loss:", float(aux.aux_loss))

    forced_idx = torch.tensor([[1, 3]]).repeat(B, 1)
    forced_weight = torch.tensor([[0.7, 0.3]], dtype=z.dtype).repeat(B, 1)
    z = conditioner(obs, topk_idx=forced_idx, topk_weight=forced_weight)
    print("forced z:", z.shape)
