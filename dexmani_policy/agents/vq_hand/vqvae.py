"""
VqVaeHand — VQ-VAE for hand pose discretisation.

Architecture:
  hand_pose (B, hand_dim) → EncoderMLP → latent (B, latent_dim)
    → ResidualVQ (2 groups × 4 codes = 16 combinations) → quantized
    → EncoderMLP (reused as decoder) → reconstructed hand_pose (B, hand_dim)

Loss:
  L = L1_recon(per-finger weighted) × 3 + commitment_loss × 5

Ported from DQ-RISE and adapted for DexMani:
  - Single-step quantization only (no action chunks)
  - No built-in optimizer (training script manages it)
  - Hand dim and codebook sizes fully configurable
  - Orthogonal initialisation for encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_vq import ResidualVQ


# ---------------------------------------------------------------------------
# Encoder / Decoder MLP
# ---------------------------------------------------------------------------

def _orthogonal_init(m):
    """Orthogonal initialisation for linear layers (DQ-RISE convention)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class EncoderMLP(nn.Module):
    """MLP: input_dim → hidden_dim → ... → hidden_dim → output_dim.

    Uses orthogonal weight init and ReLU activations.  Reused for both
    encoder and decoder (matching original DQ-RISE).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.apply(_orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


# ---------------------------------------------------------------------------
# VqVaeHand
# ---------------------------------------------------------------------------

class VqVaeHand(nn.Module):
    """VQ-VAE for single-step hand-pose discretisation.

    Encoder:  hand_dim → hidden_dim → ... → latent_dim
    VQ:       ResidualVQ (num_groups layers, codebook_size codes each)
    Decoder:  latent_dim → hidden_dim → ... → hand_dim

    Total discrete combinations: codebook_size ** num_groups
    (default: 4² = 16).

    Parameters
    ----------
    hand_dim:
        Dimensionality of the hand joint state (e.g. 6 for DQ-RISE, 10 for DexMani).
    latent_dim:
        Bottleneck dimension before/after VQ.  Default 256.
    hidden_dim:
        Hidden dimension for encoder and decoder MLPs.  Default 512.
    num_groups:
        Number of residual VQ layers (codebook groups).  Default 2.
    codebook_size:
        Number of codes per group.  Default 4.
    num_layers:
        Number of hidden layers in the encoder / decoder MLP trunk.  Default 1
        (matches original DQ-RISE ``layer_num=1``).
    act_scale:
        Scale factor applied to actions before encoding (÷) and after decoding (×).
        Default 1.0 (no scaling — data should already be in [-1, 1]).
    loss_weight:
        Per-dimension weight for the L1 reconstruction loss (required).
        Must have length ``hand_dim``.
    vq_decay:
        EMA decay for codebook updates.
    threshold_ema_dead_code:
        Cluster-size threshold for dead-code replacement.  0 = disabled.
    kmeans_init:
        If True, initialise codebook embeddings via k-means on the first batch.
    kmeans_iters:
        Number of k-means iterations (only used when ``kmeans_init=True``).
    """

    def __init__(
        self,
        hand_dim: int,
        loss_weight: list[float],  # required — no default to prevent silent mismatch
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
    ):
        super().__init__()
        self.hand_dim = hand_dim
        self.latent_dim = latent_dim
        self.register_buffer(
            'act_scale', torch.tensor(act_scale, dtype=torch.float32),
        )

        # Encoder / decoder
        self.encoder = EncoderMLP(hand_dim, latent_dim, hidden_dim, num_layers)
        self.decoder = EncoderMLP(latent_dim, hand_dim, hidden_dim, num_layers)

        # Residual VQ: num_groups layers × codebook_size codes each
        self.vq_layer = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_groups,
            codebook_size=codebook_size,
            codebook_dim=latent_dim,
            decay=vq_decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            ema_update=True,
            learnable_codebook=False,
        )
        self.num_groups = num_groups
        self.codebook_size = codebook_size

        # Per-finger / per-dimension L1 reconstruction weight
        self.register_buffer('loss_weight', torch.tensor(loss_weight, dtype=torch.float32))
        if len(loss_weight) != hand_dim:
            raise ValueError(
                f'loss_weight length ({len(loss_weight)}) must match hand_dim ({hand_dim})'
            )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, hand_pose: torch.Tensor):
        """Forward pass — encode, quantize, decode, compute losses.

        Matches original DQ-RISE VqVae.forward():
          - L1 reconstruction loss computed in scaled space (state/act_scale).
          - Decoder outputs raw values (no act_scale multiplication).
          - Commitment loss is the raw sum of per-layer losses (no extra weight).

        Args:
            hand_pose:  (B, hand_dim)  normalised to [-1, 1]

        Returns
        -------
        encoder_loss : scalar
            Per-finger-weighted L1 reconstruction loss.
        vq_loss : scalar
            Sum of per-layer commitment losses.
        vq_indices : (B, num_groups)
            Discrete codebook indices per sample.
        recon_mse : scalar
            Unweighted MSE reconstruction loss (for logging only).
        """
        # Scale & encode  (state / act_scale → encoder)
        x = hand_pose / self.act_scale                # (B, hand_dim)
        z = self.encoder(x)                           # (B, latent_dim)
        z_flat = z.unsqueeze(1)                        # (B, 1, latent_dim)

        # Residual VQ
        z_q, indices, vq_losses = self.vq_layer(z_flat)
        # z_q:     (B, 1, latent_dim)
        # indices: (B, 1, num_groups)
        # vq_losses: (1, num_groups)

        z_q = z_q.squeeze(1)                           # (B, latent_dim)
        vq_indices = indices.squeeze(1)                 # (B, num_groups)

        # Decode — raw output (NO act_scale), matching original.
        # The original computes L1 in scaled space: |state/act_scale - decoder(z_q)|.
        dec_out = self.decoder(z_q)                     # (B, hand_dim)

        # Per-dimension weighted L1 (in scaled space)
        diff = (x - dec_out).abs()                      # (B, hand_dim)
        encoder_loss = (diff * self.loss_weight).mean()

        # Unweighted MSE (for logging, in scaled space matching original)
        recon_mse = F.mse_loss(x, dec_out)

        # Commitment loss: raw sum (no extra multiplier — each VQ layer
        # already has commitment_weight=1.0).
        vq_loss = vq_losses.sum()

        return encoder_loss, vq_loss, vq_indices, recon_mse

    # ------------------------------------------------------------------
    # encode / decode helpers (used during codebook extraction & training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_to_index(self, hand_pose: torch.Tensor) -> torch.Tensor:
        """Encode hand pose → VQ indices (for training-label generation).

        Args:
            hand_pose:  (B, hand_dim)
        Returns:
            indices:    (B, num_groups)  discrete codebook indices
        """
        x = hand_pose / self.act_scale
        z = self.encoder(x)
        _, indices, _ = self.vq_layer(z.unsqueeze(1))
        return indices.squeeze(1)

    @torch.no_grad()
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vector → hand pose (for codebook extraction).

        Args:
            latent:  (*, latent_dim)
        Returns:
            hand_pose:  (*, hand_dim)
        """
        return self.decoder(latent) * self.act_scale

    # ------------------------------------------------------------------
    # codebook accessors
    # ------------------------------------------------------------------

    @property
    def codebooks(self):
        """Return stacked codebooks: (num_groups, codebook_size, latent_dim)."""
        return self.vq_layer.codebooks

