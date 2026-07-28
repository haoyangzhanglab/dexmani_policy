"""VQ-VAE for single-step dexterous-hand state quantisation."""

from __future__ import annotations

import re
from collections.abc import Mapping

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
    """MLP with an explicit number of hidden linear layers.

    ``num_layers=1`` now means exactly ``input -> hidden -> output``.  The
    previous implementation accidentally created ``num_layers + 1`` hidden
    layers.  Existing checkpoints remain loadable through
    :meth:`VQVAEHand.from_checkpoint`, which infers the actual depth from the
    state dict rather than trusting legacy configuration metadata.
    """

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
                f"loss_weight length ({len(loss_weight)}) must equal "
                f"hand_dim ({hand_dim})"
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
        _, indices, _ = self.vq_layer(
            encoded.unsqueeze(1), freeze_codebook=True
        )
        return indices.squeeze(1)

    @torch.no_grad()
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        # Range validation/clamping is deliberately performed by
        # CodebookManager during prototype export so violations are reported.
        return self.decoder(latent) * self.act_scale

    @property
    def codebooks(self) -> torch.Tensor:
        return self.vq_layer.codebooks

    # ------------------------------------------------------------------
    # Robust checkpoint reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _args_to_dict(args) -> dict:
        if args is None:
            return {}
        if isinstance(args, Mapping):
            return dict(args)
        if hasattr(args, "__dict__"):
            return vars(args)
        return {}

    @staticmethod
    def _infer_hidden_layers(state: Mapping[str, torch.Tensor]) -> int:
        pattern = re.compile(r"^encoder\.trunk\.(\d+)\.weight$")
        linear_indices = [
            int(match.group(1))
            for key in state
            if (match := pattern.match(key)) is not None
        ]
        if not linear_indices:
            raise ValueError("Cannot infer encoder depth from checkpoint")
        return len(linear_indices)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "VQVAEHand":
        """Construct from either a complete training checkpoint or state dict.

        Architecture dimensions are inferred from tensor shapes.  This loads
        both corrected checkpoints and legacy checkpoints whose saved
        ``num_layers`` value was off by one.
        """
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            saved_args = cls._args_to_dict(checkpoint.get("args"))
        else:
            state = checkpoint
            saved_args = {}

        state = {key: value.to(map_location) for key, value in state.items()}
        first_weight = state["encoder.trunk.0.weight"]
        hand_dim = int(first_weight.shape[1])
        hidden_dim = int(first_weight.shape[0])
        latent_dim = int(state["encoder.head.weight"].shape[0])
        num_layers = cls._infer_hidden_layers(state)
        num_groups = int(state["vq_layer.layer_weights"].numel())

        embed_key = "vq_layer.layers.0._codebook.embed"
        if embed_key not in state:
            raise KeyError(f"Missing codebook tensor: {embed_key}")
        codebook_size = int(state[embed_key].shape[1])
        act_scale = float(state.get("act_scale", torch.tensor(1.0)).item())
        loss_weight_tensor = state.get(
            "loss_weight", torch.ones(hand_dim, dtype=torch.float32)
        )
        loss_weight = loss_weight_tensor.detach().cpu().flatten().tolist()

        model = cls(
            hand_dim=hand_dim,
            loss_weight=loss_weight,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            codebook_size=codebook_size,
            num_layers=num_layers,
            act_scale=act_scale,
            vq_decay=float(saved_args.get("vq_decay", 0.8)),
            threshold_ema_dead_code=int(
                saved_args.get("threshold_ema_dead_code", 0)
            ),
            # The saved EMA buffers and embeddings are loaded below, so no
            # fresh k-means initialisation is needed.
            kmeans_init=False,
            kmeans_iters=int(saved_args.get("kmeans_iters", 10)),
        )
        model.load_state_dict(state, strict=strict)
        return model
