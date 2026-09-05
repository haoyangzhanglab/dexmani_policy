"""VQ-VAE for single-step dexterous-hand state quantisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_vq import ResidualVQ


def _orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class EncoderMLP(nn.Module):
    """MLP with an explicit number of hidden linear layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.apply(_orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class VQVAEHand(nn.Module):
    """Residual VQ-VAE for a single hand pose."""

    def __init__(
        self,
        hand_dim: int,
        loss_weight: list[float],
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_groups: int = 2,
        codebook_size: int = 4,
        num_layers: int = 1,
        act_scale: float = 1.0,
        vq_decay: float = 0.8,
        threshold_ema_dead_code: int = 0,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
    ) -> None:
        super().__init__()
        if len(loss_weight) != hand_dim:
            raise ValueError(
                f"loss_weight length ({len(loss_weight)}) must equal hand_dim ({hand_dim})"
            )

        self.hand_dim = int(hand_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_groups = int(num_groups)
        self.codebook_size = int(codebook_size)
        self.num_layers = int(num_layers)

        self.register_buffer(
            "act_scale", torch.tensor(float(act_scale), dtype=torch.float32)
        )
        self.register_buffer(
            "loss_weight", torch.tensor(loss_weight, dtype=torch.float32)
        )

        self.encoder = EncoderMLP(
            self.hand_dim, self.latent_dim, self.hidden_dim, self.num_layers
        )
        self.decoder = EncoderMLP(
            self.latent_dim, self.hand_dim, self.hidden_dim, self.num_layers
        )
        self.vq_layer = ResidualVQ(
            dim=self.latent_dim,
            num_quantizers=self.num_groups,
            codebook_size=self.codebook_size,
            codebook_dim=self.latent_dim,
            decay=vq_decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            ema_update=True,
            learnable_codebook=False,
        )

    def forward(self, hand_pose: torch.Tensor):
        if hand_pose.shape[-1] != self.hand_dim:
            raise ValueError(
                f"Expected hand_dim={self.hand_dim}, got {hand_pose.shape[-1]}"
            )
        x = hand_pose / self.act_scale
        encoded = self.encoder(x)
        quantized, indices, vq_losses = self.vq_layer(encoded.unsqueeze(1))
        decoded = self.decoder(quantized.squeeze(1))

        weighted_l1 = ((x - decoded).abs() * self.loss_weight).mean()
        reconstruction_mse = F.mse_loss(x, decoded)
        commitment_loss = vq_losses.sum()
        return (
            weighted_l1,
            commitment_loss,
            indices.squeeze(1),
            reconstruction_mse,
        )

    @torch.no_grad()
    def encode_to_index(self, hand_pose: torch.Tensor) -> torch.Tensor:
        """Encode without changing EMA codebook state, even in train mode."""
        x = hand_pose / self.act_scale
        encoded = self.encoder(x)
        _, indices, _ = self.vq_layer(encoded.unsqueeze(1), freeze_codebook=True)
        return indices.squeeze(1)

    @torch.no_grad()
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        # Range validation/clamping is deliberately performed by
        # CodebookManager during prototype export so violations are reported.
        return self.decoder(latent) * self.act_scale

    @property
    def codebooks(self) -> torch.Tensor:
        return self.vq_layer.codebooks

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "VQVAEHand":
        """Construct from the current VQ training checkpoint schema."""
        if checkpoint.get("format_version") != 3:
            raise ValueError("VQ checkpoint must use format_version=3")
        config = checkpoint["model_config"]
        state = {
            key: value.to(map_location)
            for key, value in checkpoint["model_state_dict"].items()
        }

        model = cls(
            hand_dim=int(config["hand_dim"]),
            loss_weight=list(config["loss_weight"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_groups=int(config["num_groups"]),
            codebook_size=int(config["codebook_size"]),
            num_layers=int(config["num_layers"]),
            act_scale=float(config["act_scale"]),
            vq_decay=float(config["vq_decay"]),
            threshold_ema_dead_code=int(config["threshold_ema_dead_code"]),
            kmeans_init=False,
            kmeans_iters=int(config["kmeans_iters"]),
        )
        model.load_state_dict(state, strict=True)
        return model
