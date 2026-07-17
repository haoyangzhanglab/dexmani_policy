"""
ResidualVQ + GroupedResidualVQ — residual vector quantization.

Follows Algorithm 1 from https://arxiv.org/pdf/2107.03312.pdf

Ported from DQ-RISE.  Simplifications vs. original:
  - Removed: accept_image_fmap, shared_codebook, quantize_dropout
    (not needed for single-step, small-codebook use case)
  - Kept: layer_weights with softmax combination, CE loss path for training,
    GroupedResidualVQ (splits feature dim into independent VQ groups)
"""

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from .vector_quantize import VectorQuantize, exists, default


class ResidualVQ(nn.Module):
    """Residual Vector Quantization — cascaded VQ layers."""

    def __init__(
        self,
        *,
        dim: int,
        num_quantizers: int,       # number of residual layers
        codebook_size: int = 4,
        codebook_dim: int | None = None,
        decay: float = 0.8,
        eps: float = 1e-5,
        ema_warmup_steps: int = 0,
        threshold_ema_dead_code: int = 0,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sample_codebook_temp: float = 1.,
        ema_update: bool = True,
        learnable_codebook: bool = False,
    ):
        super().__init__()
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        # Learnable layer weights (initialised [0.5, 0.5], softmax-normalised)
        self.layer_weights = nn.Parameter(
            torch.full((num_quantizers,), 0.5, dtype=torch.float32)
        )

        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([
            VectorQuantize(
                dim=codebook_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                decay=decay,
                eps=eps,
                ema_warmup_steps=ema_warmup_steps,
                threshold_ema_dead_code=threshold_ema_dead_code,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sample_codebook_temp=sample_codebook_temp,
                ema_update=ema_update,
                learnable_codebook=learnable_codebook,
                commitment_weight=1.0,   # commitment loss collected per layer
            )
            for _ in range(num_quantizers)
        ])

    @property
    def codebooks(self):
        """Return stacked codebooks: (num_quantizers, codebook_size, codebook_dim)."""
        cbs = [layer._codebook.embed for layer in self.layers]
        return rearrange(torch.stack(cbs, dim=0), 'q 1 c d -> q c d')

    def get_codes_from_indices(self, indices):
        """
        Reconstruct quantized vector from per-layer indices.
        Args:
            indices: (B, num_quantizers)  — one index per layer
        Returns:
            codes: (num_quantizers, B, codebook_dim)  — per-layer code vectors
        """
        batch = indices.shape[0]
        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b=batch)   # (Q, B, C, D)
        gather_indices = repeat(indices, 'b q -> q b 1 d', d=codebooks.shape[-1])
        all_codes = codebooks.gather(2, gather_indices)                    # (Q, B, 1, D)
        return all_codes.squeeze(2)                                        # (Q, B, D)

    def forward(self, x, indices=None, sample_codebook_temp=None):
        """
        Args:
            x:        (B, N, dim)  input to quantize
            indices:  (B, N, num_quantizers)  if given, compute CE loss
        Returns:
            quantized_out:  (B, N, dim)  reconstructed after projection
            all_indices:    (B, N, num_quantizers)  per-layer code indices
            all_losses:     (num_quantizers,)  per-layer commitment losses
        """
        return_loss = exists(indices)
        device = x.device
        num_quant = self.num_quantizers

        x = self.project_in(x)

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            ce_losses = []

        for quantizer_index, layer in enumerate(self.layers):
            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            quantized, *rest = layer(
                residual,
                indices=layer_indices,
                sample_codebook_temp=sample_codebook_temp,
            )

            # Residual: subtract the *detached* quantized to isolate next layer's job
            residual = residual - quantized.detach()

            # Softmax-weighted contribution to output
            weight = F.softmax(self.layer_weights, dim=0)[quantizer_index]
            quantized_out = quantized_out + quantized * weight

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)

        quantized_out = self.project_out(quantized_out)

        if return_loss:
            return quantized_out, sum(ce_losses)

        all_indices = torch.stack(all_indices, dim=-1)    # (B, N, num_quantizers)
        all_losses = torch.stack(all_losses, dim=-1)       # (1, num_quantizers)

        return quantized_out, all_indices, all_losses


# ===========================================================================
# GroupedResidualVQ — splits feature dim into independent ResidualVQ groups
# ===========================================================================

class GroupedResidualVQ(nn.Module):
    """Grouped Residual Vector Quantization.

    Splits the feature dimension into ``groups`` independent chunks, each
    quantized by its own :class:`ResidualVQ` stack.  Outputs are concatenated
    back along the feature axis.

    Ported from DQ-RISE ``policy/vqvae_rise/vector_quantize_pytorch/residual_vq.py``
    """

    def __init__(self, *, dim: int, groups: int = 1, **kwargs):
        super().__init__()
        self.dim = dim
        self.groups = groups
        if dim % groups != 0:
            raise ValueError(
                f'dim ({dim}) must be divisible by groups ({groups})'
            )
        dim_per_group = dim // groups

        self.rvqs = nn.ModuleList([
            ResidualVQ(dim=dim_per_group, **kwargs)
            for _ in range(groups)
        ])

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    def get_codes_from_indices(self, indices):
        codes = tuple(
            rvq.get_codes_from_indices(chunk_indices)
            for rvq, chunk_indices in zip(self.rvqs, indices)
        )
        return torch.stack(codes)

    def forward(self, x, indices=None, sample_codebook_temp=None):
        shape, split_dim = x.shape, -1
        x_chunks = x.chunk(self.groups, dim=split_dim)

        indices_chunks = (indices,) if indices is not None else ((),)
        if len(indices_chunks) > 0 and len(indices_chunks[0]) > 0:
            indices_chunks = indices.chunk(self.groups, dim=-1)
        else:
            indices_chunks = [None] * self.groups

        out = tuple(
            rvq(chunk, indices=chunk_indices,
                sample_codebook_temp=sample_codebook_temp)
            for rvq, chunk, chunk_indices in
            zip(self.rvqs, x_chunks, indices_chunks)
        )
        out = tuple(zip(*out))

        # If CE loss path: quantized + sum of losses
        if indices is not None:
            quantized, ce_losses = out
            return torch.cat(quantized, dim=split_dim), sum(ce_losses)

        # Normal path: quantized + all_indices + all_losses
        quantized, all_indices, commit_losses = out
        quantized = torch.cat(quantized, dim=split_dim)
        all_indices = torch.stack(all_indices)       # (groups, B, N, num_quantizers)
        commit_losses = torch.stack(commit_losses)   # (groups, 1, num_quantizers)

        return quantized, all_indices, commit_losses
