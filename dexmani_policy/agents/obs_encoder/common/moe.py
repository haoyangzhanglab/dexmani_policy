from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# MoE-DP paper-aligned defaults
# -----------------------------------------------------------------------------
# The paper does not publish a single universal configuration for every task.
# From Table 3, the most common setting across both simulation and real tasks is:
#   #experts = 16
#   top-k     = 2
#   lambda    = 0.1   (load-balancing loss weight)
#   beta      = 0.01  (entropy loss weight; common simulation default)
# We therefore expose these as the default "paper" configuration.
PAPER_NUM_EXPERTS = 16
PAPER_TOP_K = 2
PAPER_LOAD_BALANCE_COEF = 0.1
PAPER_ENTROPY_COEF = 0.01
PAPER_ROUTER_TEMPERATURE = 1.0


@dataclass
class MoEAuxOutput:
    """Auxiliary outputs needed for analysis/training."""

    router_logits: torch.Tensor
    router_probs: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    load_balance_loss: torch.Tensor
    entropy_loss: torch.Tensor
    aux_loss: torch.Tensor


class MLPExpert(nn.Module):
    """A small MLP expert used inside the MoE layer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        prev = in_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseTopKMoE(nn.Module):
    """
    A clean Top-k sparse MoE layer.

    This follows the MoE-DP design at a high level:
      - router = softmax(W_g x)
      - select top-k experts
      - output = sum_i g_i * E_i(x) over selected experts
      - auxiliary losses = load-balance + entropy

    Notes
    -----
    * This implementation is intentionally clean/readable rather than maximally optimized.
    * Input shape can be [B, C] or [..., C]; all leading dims are treated as tokens.
    * Manual expert override is supported for inference-time control.
    * Defaults are aligned with the most common settings reported in MoE-DP Table 3.
    """

    def __init__(
        self,
        in_dim: int,
        num_experts: int = PAPER_NUM_EXPERTS,
        top_k: int = PAPER_TOP_K,
        expert_hidden_dim: int = 256,
        expert_out_dim: Optional[int] = None,
        expert_num_layers: int = 2,
        router_temperature: float = PAPER_ROUTER_TEMPERATURE,
        load_balance_coef: float = PAPER_LOAD_BALANCE_COEF,
        entropy_coef: float = PAPER_ENTROPY_COEF,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        use_router_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")

        self.in_dim = in_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_dim = in_dim if expert_out_dim is None else expert_out_dim
        self.router_temperature = router_temperature
        self.load_balance_coef = load_balance_coef
        self.entropy_coef = entropy_coef

        self.router = nn.Linear(in_dim, num_experts, bias=use_router_bias)
        self.experts = nn.ModuleList(
            [
                MLPExpert(
                    in_dim=in_dim,
                    hidden_dim=expert_hidden_dim,
                    out_dim=self.out_dim,
                    num_layers=expert_num_layers,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

    def _flatten_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        if x.ndim < 2:
            raise ValueError("Expected input shape [B, C] or [..., C]")
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        return x_flat, original_shape

    def _unflatten_tokens(self, x_flat: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        return x_flat.reshape(*original_shape, x_flat.shape[-1])

    def _compute_aux_losses(
        self, router_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # MoE-DP defines:
        #   f_i = fraction of samples dispatched to expert i (via argmax)
        #   P_i = average router probability for expert i
        #   L_load = N * sum_i f_i * P_i
        #   L_entropy = mean_t [ - sum_i p_t,i log p_t,i ]
        # We minimize lambda * L_load + beta * L_entropy, matching the paper.
        hard_assign = router_probs.argmax(dim=-1)  # [T]
        f_i = F.one_hot(hard_assign, num_classes=self.num_experts).float().mean(dim=0)
        p_i = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(f_i * p_i)

        entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-9))).sum(dim=-1).mean()
        aux_loss = self.load_balance_coef * load_balance_loss + self.entropy_coef * entropy
        return load_balance_loss, entropy, aux_loss

    def forward(
        self,
        x: torch.Tensor,
        manual_topk_indices: Optional[torch.Tensor] = None,
        manual_topk_weights: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[MoEAuxOutput]]:
        x_flat, original_shape = self._flatten_tokens(x)

        router_logits = self.router(x_flat) / max(self.router_temperature, 1e-6)
        router_probs = torch.softmax(router_logits, dim=-1)

        # Default autonomous routing.
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # Optional inference-time override.
        if manual_topk_indices is not None:
            if manual_topk_indices.shape != topk_indices.shape:
                raise ValueError(
                    f"manual_topk_indices must have shape {tuple(topk_indices.shape)}, got {tuple(manual_topk_indices.shape)}"
                )
            topk_indices = manual_topk_indices
            if manual_topk_weights is None:
                topk_weights = torch.gather(router_probs, dim=-1, index=topk_indices)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            else:
                if manual_topk_weights.shape != topk_weights.shape:
                    raise ValueError(
                        f"manual_topk_weights must have shape {tuple(topk_weights.shape)}, got {tuple(manual_topk_weights.shape)}"
                    )
                topk_weights = manual_topk_weights
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Evaluate all experts once for readability; then gather only selected outputs.
        # expert_outputs: [T, E, D]
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)

        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, self.out_dim)  # [T, K, D]
        selected_outputs = torch.gather(expert_outputs, dim=1, index=gather_index)  # [T, K, D]
        y_flat = torch.sum(selected_outputs * topk_weights.unsqueeze(-1), dim=1)  # [T, D]
        y = self._unflatten_tokens(y_flat, original_shape)

        aux: Optional[MoEAuxOutput] = None
        if return_aux:
            load_loss, entropy_loss, aux_loss = self._compute_aux_losses(router_probs)
            aux = MoEAuxOutput(
                router_logits=self._unflatten_tokens(router_logits, original_shape),
                router_probs=self._unflatten_tokens(router_probs, original_shape),
                topk_indices=topk_indices.reshape(*original_shape, self.top_k),
                topk_weights=topk_weights.reshape(*original_shape, self.top_k),
                load_balance_loss=load_loss,
                entropy_loss=entropy_loss,
                aux_loss=aux_loss,
            )

        return y, aux

    @torch.no_grad()
    def get_primary_expert(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax(router(x)) for debugging / visualization."""
        x_flat, original_shape = self._flatten_tokens(x)
        router_probs = torch.softmax(self.router(x_flat), dim=-1)
        expert_idx = router_probs.argmax(dim=-1)
        return expert_idx.reshape(*original_shape)



from typing import Dict, Tuple
from dexmani_policy.agents.obs_encoder import DP3Encoder

class DP3MoEConditioner(nn.Module):
    """
    Wrap a DP3Encoder with an MoE layer.

    This mirrors the MoE-DP paper's placement of the MoE block:
        encoder -> MoE -> diffusion model condition

    Since your current codebase already has a DP3-style encoder, this module only changes the
    conditioning path and leaves the downstream diffusion policy head untouched.
    """

    def __init__(
        self,
        dp3_encoder: DP3Encoder,
        num_experts: int = PAPER_NUM_EXPERTS,
        top_k: int = PAPER_TOP_K,
        expert_hidden_dim: int = 256,
        expert_out_dim: Optional[int] = None,
        expert_num_layers: int = 2,
        load_balance_coef: float = PAPER_LOAD_BALANCE_COEF,
        entropy_coef: float = PAPER_ENTROPY_COEF,
        router_temperature: float = PAPER_ROUTER_TEMPERATURE,
    ) -> None:
        super().__init__()
        self.encoder = dp3_encoder
        self.moe = SparseTopKMoE(
            in_dim=self.encoder.out_shape,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            expert_out_dim=expert_out_dim if expert_out_dim is not None else self.encoder.out_shape,
            expert_num_layers=expert_num_layers,
            load_balance_coef=load_balance_coef,
            entropy_coef=entropy_coef,
            router_temperature=router_temperature,
        )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        manual_topk_indices: Optional[torch.Tensor] = None,
        manual_topk_weights: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[MoEAuxOutput]]:
        # Base DP3 encoder outputs the fused point-cloud + joint-state feature.
        z = self.encoder(observations)  # [B, C]
        z_prime, aux = self.moe(
            z,
            manual_topk_indices=manual_topk_indices,
            manual_topk_weights=manual_topk_weights,
            return_aux=return_aux,
        )
        return z_prime, aux

    @property
    def out_shape(self) -> int:
        return self.moe.out_dim