"""Single-step hand-state quantization adapted from RISE-Policy/DQ-RISE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dexmani_policy.utils.validation import positive_int


def orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        nn.init.zeros_(module.bias)


def sample_vectors(samples: torch.Tensor, count: int) -> torch.Tensor:
    size = len(samples)
    if size >= count:
        indices = torch.randperm(size, device=samples.device)[:count]
    else:
        indices = torch.randint(size, (count,), device=samples.device)
    return samples[indices]


@torch.no_grad()
def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int):
    means = sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        indices = torch.cdist(samples, means).argmin(-1)
        counts = torch.bincount(indices, minlength=num_clusters)
        sums = torch.zeros_like(means)
        sums.scatter_add_(0, indices[:, None].expand_as(samples), samples)
        centers = sums / counts.clamp_min(1)[:, None]
        means = torch.where(counts[:, None] == 0, means, centers)
    return means, counts


class HandMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        positive_int(num_layers, "num_layers")
        layers = []
        for _ in range(num_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.head(self.trunk(x))


class EMAVectorQuantizer(nn.Module):
    """Euclidean nearest-neighbor quantization with EMA codebook updates."""

    def __init__(
        self, dim, codebook_size, decay=0.8, eps=1e-5,
        kmeans_init=False, kmeans_iters=10, threshold_ema_dead_code=0,
    ):
        super().__init__()
        self.codebook_size = positive_int(codebook_size, "codebook_size")
        self.kmeans_iters = positive_int(kmeans_iters, "kmeans_iters")
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        embed = torch.zeros(codebook_size, dim)
        if not kmeans_init:
            # Preserve DQ-RISE's fan-in for its original (1, K, D) embedding.
            nn.init.kaiming_uniform_(embed.unsqueeze(0))
        self.register_buffer("initialized", torch.tensor(not kmeans_init))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.zeros(codebook_size))

    @torch.no_grad()
    def initialize(self, samples):
        embed, counts = kmeans(samples, self.codebook_size, self.kmeans_iters)
        self.embed.copy_(embed)
        self.embed_avg.copy_(embed * counts[:, None])
        self.cluster_size.copy_(counts)
        self.initialized.fill_(True)

    @torch.no_grad()
    def update_codebook(self, samples, indices):
        assignments = F.one_hot(indices, self.codebook_size).to(samples.dtype)
        self.cluster_size.lerp_(assignments.sum(0), 1 - self.decay)
        self.embed_avg.lerp_(assignments.T @ samples, 1 - self.decay)
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

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        samples = x.reshape(-1, x.shape[-1]).float()
        if not self.initialized:
            if not self.training:
                raise RuntimeError("Initialize the codebook in training before evaluation")
            self.initialize(samples)

        distances = (
            samples.square().sum(-1, keepdim=True)
            + self.embed.square().sum(-1)[None]
            - 2 * (samples @ self.embed.T)
        ).clamp_min(1e-12).sqrt()
        indices = distances.argmin(-1)
        quantized = F.embedding(indices, self.embed).reshape_as(x)
        commitment = F.mse_loss(quantized.detach(), x.float())
        if self.training:
            self.update_codebook(samples, indices)
            quantized = x + (quantized - x).detach()
        return quantized, indices.reshape(x.shape[:-1]), commitment


class ResidualVectorQuantizer(nn.Module):
    """Residual quantization with learned, softmax-normalized layer weights."""

    def __init__(self, dim, num_quantizers, codebook_size, **kwargs):
        super().__init__()
        positive_int(num_quantizers, "num_quantizers")
        self.layer_weights = nn.Parameter(torch.full((num_quantizers,), 0.5))
        self.layers = nn.ModuleList([
            EMAVectorQuantizer(dim, codebook_size, **kwargs)
            for _ in range(num_quantizers)
        ])

    @property
    def codebooks(self):
        return torch.stack([layer.embed for layer in self.layers])

    def forward(self, x):
        residual, output = x, 0.0
        indices, losses = [], []
        weights = self.layer_weights.softmax(0)
        for weight, layer in zip(weights, self.layers):
            quantized, index, loss = layer(residual)
            residual = residual - quantized.detach()
            output = output + weight * quantized
            indices.append(index)
            losses.append(loss)
        return output, torch.stack(indices, -1), torch.stack(losses)


class HandVQVAE(nn.Module):
    """Residual VQ-VAE for individual hand states, independent of action chunks."""

    FORMAT_VERSION = 4

    def __init__(
        self, hand_dim, loss_weight, latent_dim=256, hidden_dim=512,
        num_groups=2, codebook_size=4, num_layers=1, act_scale=1.0,
        vq_decay=0.8, threshold_ema_dead_code=0, kmeans_init=False, kmeans_iters=10,
    ):
        super().__init__()
        if len(loss_weight) != hand_dim:
            raise ValueError("loss_weight must contain one weight per hand joint")
        if not 0 < act_scale < float("inf"):
            raise ValueError("act_scale must be positive and finite")
        self.hand_dim = hand_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))
        self.register_buffer("loss_weight", torch.tensor(loss_weight, dtype=torch.float32))
        self.encoder = HandMLP(hand_dim, latent_dim, hidden_dim, num_layers)
        self.decoder = HandMLP(latent_dim, hand_dim, hidden_dim, num_layers)
        self.quantizer = ResidualVectorQuantizer(
            latent_dim, num_groups, codebook_size, decay=vq_decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
        )

    def forward(self, hand):
        target = hand / self.act_scale
        quantized, indices, losses = self.quantizer(self.encoder(target))
        reconstruction = self.decoder(quantized)
        reconstruction_loss = ((target - reconstruction).abs() * self.loss_weight).mean()
        mse = F.mse_loss(target, reconstruction)
        return reconstruction_loss, losses.sum(), indices, mse

    def decode(self, latent):
        return self.decoder(latent) * self.act_scale

    @property
    def codebooks(self):
        return self.quantizer.codebooks

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu"):
        if checkpoint["format_version"] != cls.FORMAT_VERSION:
            raise ValueError(f"HandVQVAE requires checkpoint format_version={cls.FORMAT_VERSION}")
        model = cls(**checkpoint["model_config"], kmeans_init=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model
