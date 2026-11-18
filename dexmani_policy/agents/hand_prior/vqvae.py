"""Single-step hand-state quantization adapted from RISE-Policy/DQ-RISE."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dexmani_policy.agents.hand_prior.codebook import codebook_distances
from dexmani_policy.agents.modules import build_mlp
from dexmani_policy.common.config import positive_int


def orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        nn.init.zeros_(module.bias)


def sample_vectors(samples: Tensor, count: int) -> Tensor:
    size = len(samples)
    if size >= count:
        indices = torch.randperm(size, device=samples.device)[:count]
    else:
        indices = torch.randint(size, (count,), device=samples.device)
    return samples[indices]


class EMAVectorQuantizer(nn.Module):
    """Euclidean nearest-neighbor quantization with EMA codebook updates."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.8,
        eps: float = 1e-5,
        threshold_ema_dead_code: float = 0,
    ) -> None:
        super().__init__()
        self.codebook_size = positive_int(codebook_size, "codebook_size")
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # Preserve DQ-RISE's fan-in for its original (1, K, D) embedding.
        embed = nn.init.kaiming_uniform_(torch.empty(1, codebook_size, dim)).squeeze(0)
        self.register_buffer("embed", embed)
        # Random centers and their accumulated sums use the same unit pseudocount.
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(codebook_size))

    @torch.no_grad()
    def update_codebook(self, samples: Tensor, indices: Tensor) -> None:
        counts = samples.new_zeros(self.codebook_size).scatter_add_(
            0, indices, samples.new_ones(len(samples)),
        )
        sums = torch.zeros_like(self.embed_avg).index_add_(0, indices, samples)
        self.cluster_size.lerp_(counts, 1 - self.decay)
        self.embed_avg.lerp_(sums, 1 - self.decay)
        total = self.cluster_size.sum()
        counts = (self.cluster_size + self.eps) / (total + self.codebook_size * self.eps) * total
        self.embed.copy_(self.embed_avg / counts[:, None])

        if self.threshold_ema_dead_code:
            expired = self.cluster_size < self.threshold_ema_dead_code
            count = int(expired.sum())
            if count:
                replacements = sample_vectors(samples, count)
                self.embed[expired] = replacements
                self.cluster_size[expired] = self.threshold_ema_dead_code
                self.embed_avg[expired] = replacements * self.threshold_ema_dead_code

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            samples = x.detach().reshape(-1, x.shape[-1]).float()
            distances = codebook_distances(samples, self.embed)
            indices = distances.argmin(-1)
            quantized = F.embedding(indices, self.embed).reshape_as(x)
            commitment = F.mse_loss(quantized.detach(), x.float())
            if self.training:
                self.update_codebook(samples, indices)
                quantized = x + (quantized - x).detach()

        return quantized, commitment


class ResidualVectorQuantizer(nn.Module):
    """Residual quantization with learned, softmax-normalized layer weights."""

    def __init__(self, dim: int, num_quantizers: int, codebook_size: int, **kwargs) -> None:
        super().__init__()
        positive_int(num_quantizers, "num_quantizers")
        self.layer_weights = nn.Parameter(torch.full((num_quantizers,), 0.5))
        self.layers = nn.ModuleList(
            [EMAVectorQuantizer(dim, codebook_size, **kwargs) for _ in range(num_quantizers)]
        )

    @property
    def codebooks(self) -> Tensor:
        return torch.stack([layer.embed for layer in self.layers])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        residual, output = x, 0.0
        losses = []
        weights = self.layer_weights.softmax(0)
        for weight, layer in zip(weights, self.layers):
            quantized, loss = layer(residual)
            residual = residual - quantized.detach()
            output = output + weight * quantized
            losses.append(loss)
        return cast(Tensor, output), torch.stack(losses).sum()


class HandVQVAE(nn.Module):
    """Residual VQ-VAE for individual hand states, independent of action chunks."""

    def __init__(
        self,
        hand_dim: int,
        loss_weight: Sequence[float],
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_groups: int = 2,
        codebook_size: int = 4,
        num_layers: int = 6,
        vq_decay: float = 0.8,
        threshold_ema_dead_code: float = 0,
    ) -> None:
        super().__init__()
        if len(loss_weight) != hand_dim:
            raise ValueError("loss_weight must contain one weight per hand joint")
        positive_int(num_layers, "num_layers")

        self.hand_dim = hand_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.register_buffer("loss_weight", torch.tensor(loss_weight, dtype=torch.float32))
        self.encoder = build_mlp(
            [hand_dim] + [hidden_dim] * num_layers + [latent_dim],
            activation=nn.ReLU, normalize=False,
        )
        self.encoder.apply(orthogonal_init)
        self.decoder = build_mlp(
            [latent_dim] + [hidden_dim] * num_layers + [hand_dim],
            activation=nn.ReLU, normalize=False,
        )
        self.decoder.apply(orthogonal_init)
        self.quantizer = ResidualVectorQuantizer(
            latent_dim,
            num_groups,
            codebook_size,
            decay=vq_decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

    @property
    def model_config(self) -> dict:
        layer = self.quantizer.layers[0]
        return {
            "hand_dim": self.hand_dim, "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim, "num_groups": self.num_groups,
            "codebook_size": self.codebook_size, "num_layers": self.num_layers,
            "loss_weight": self.loss_weight.detach().cpu().tolist(),
            "vq_decay": layer.decay,
            "threshold_ema_dead_code": layer.threshold_ema_dead_code,
        }

    def forward(self, hand: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        quantized, commitment = self.quantizer(self.encoder(hand))
        reconstruction = self.decoder(quantized)
        reconstruction_loss = ((hand - reconstruction).abs() * self.loss_weight).mean()
        with torch.no_grad():
            mse = F.mse_loss(hand, reconstruction)
        return reconstruction_loss, commitment, mse

    @property
    def codebooks(self) -> Tensor:
        return self.quantizer.codebooks

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, *, device: str | torch.device = "cpu") -> HandVQVAE:
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.to(device)
