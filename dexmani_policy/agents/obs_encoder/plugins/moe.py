import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MoEAux:
    router_logits: torch.Tensor
    router_probs: torch.Tensor
    topk_idx: torch.Tensor
    topk_weight: torch.Tensor
    f_i: torch.Tensor
    p_i: torch.Tensor
    load_balance_loss: torch.Tensor
    entropy_loss: torch.Tensor
    aux_loss: torch.Tensor


class Expert(nn.Module):
    """单个 expert, 保持最小 MLP 形式。"""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
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
    ) -> None:
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

    def forward(
        self,
        x: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weight: Optional[torch.Tensor] = None,
        return_aux: bool = False,
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
            topk_idx = topk_idx.reshape(-1, self.top_k).to(device=x.device, dtype=torch.long)
            if topk_weight is None:
                topk_weight = torch.gather(router_probs, dim=-1, index=topk_idx)
            else:
                topk_weight = topk_weight.reshape(-1, self.top_k).to(device=x.device, dtype=x.dtype)

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

        for i, expert in enumerate(self.experts):
            mask = expert_idx == i
            if not torch.any(mask):
                continue
            token_idx_i = token_idx[mask]
            y_i = expert(x.index_select(0, token_idx_i))
            y.index_add_(0, token_idx_i, y_i * expert_weight[mask].unsqueeze(-1))

        if self.residual:
            y = x + y

        y = y.reshape(*token_shape, self.out_dim)

        if not return_aux:
            return y

        dispatch = torch.zeros_like(router_probs)
        dispatch.scatter_(1, topk_idx, 1.0)

        f_i = dispatch.mean(dim=0) / float(self.top_k)
        p_i = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(f_i * p_i)
        entropy_loss = -(router_probs * torch.log(router_probs.clamp_min(1e-9))).sum(dim=-1).mean()
        aux_loss = self.lambda_load * load_balance_loss + self.beta_entropy * entropy_loss

        aux = MoEAux(
            router_logits=router_logits.reshape(*token_shape, self.num_experts),
            router_probs=router_probs.reshape(*token_shape, self.num_experts),
            topk_idx=topk_idx.reshape(*token_shape, self.top_k),
            topk_weight=topk_weight.reshape(*token_shape, self.top_k),
            f_i=f_i,
            p_i=p_i,
            load_balance_loss=load_balance_loss,
            entropy_loss=entropy_loss,
            aux_loss=aux_loss,
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
    ):
        z = self.encoder(obs)
        return self.moe(z, topk_idx=topk_idx, topk_weight=topk_weight, return_aux=return_aux)

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
    print("router_probs:", aux.router_probs.shape)
    print("topk_idx:", aux.topk_idx.shape)
    print("aux_loss:", float(aux.aux_loss))

    forced_idx = torch.tensor([[1, 3]]).repeat(B, 1)
    forced_weight = torch.tensor([[0.7, 0.3]], dtype=z.dtype).repeat(B, 1)
    z = conditioner(obs, topk_idx=forced_idx, topk_weight=forced_weight)
    print("forced z:", z.shape)