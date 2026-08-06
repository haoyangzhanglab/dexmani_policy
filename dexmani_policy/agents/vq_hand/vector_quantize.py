"""
VectorQuantize — core VQ layer with Euclidean codebook + EMA updates.

Simplified from DQ-RISE's vector_quantize_pytorch:
  - Removed: DDP distributed sync, CosineSimCodebook, affine_param,
    accept_image_fmap, orthogonal_reg, in_place_codebook_optimizer,
    sync_update_v, reinmax, multi-head separate_codebook_per_head
  - Kept: EuclideanCodebook with EMA updates, kmeans init, dead code
    expiration, learnable_codebook option, commitment loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from torch import einsum, nn

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def ema_inplace(old, new, decay):
    old.lerp_(new, 1 - decay)


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


# ---------------------------------------------------------------------------
# gumbel / straight-through
# ---------------------------------------------------------------------------


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(logits, temperature=1.0, stochastic=False, straight_through=False, dim=-1, training=True):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    π1 = (logits / temperature).softmax(dim=dim)
    one_hot = one_hot + π1 - π1.detach()
    return ind, one_hot


# ---------------------------------------------------------------------------
# dead-code handling
# ---------------------------------------------------------------------------


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


# ---------------------------------------------------------------------------
# k-means initialisation (single-GPU)
# ---------------------------------------------------------------------------


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    """Run batched k-means to initialise codebook embeddings."""
    num_codebooks, dim, dtype, _device = (
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )
    means = batched_sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, "h n d -> h d n")
        else:
            dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


# ===========================================================================
# EuclideanCodebook — EMA-updated codebook (single-GPU, no affine)
# ===========================================================================


class EuclideanCodebook(nn.Module):
    """Euclidean-distance codebook with EMA updates and dead-code expiry."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        num_codebooks: int = 1,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        eps: float = 1e-5,
        ema_warmup_steps: int = 0,
        threshold_ema_dead_code: int = 2,
        reset_cluster_size: int | None = None,
        learnable_codebook: bool = False,
        sample_codebook_temp: float = 1.0,
        ema_update: bool = True,
    ):
        super().__init__()

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        self.sample_codebook_temp = sample_codebook_temp

        self.register_buffer("initted", torch.tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embed.clone())

        self.ema_warmup_steps = ema_warmup_steps
        self.register_buffer("_ema_step_counter", torch.tensor(0, dtype=torch.long))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)

    # ------------------------------------------------------------------
    # k-means init
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        embed_sum = embed * rearrange(cluster_size, "... -> ... 1")
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.tensor([True]))

    # ------------------------------------------------------------------
    # dead-code replacement
    # ------------------------------------------------------------------

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue
            # samples: (N, D) — flat batch; draw 'num' random vectors from it
            sampled = sample_vectors(samples, mask.sum().item())
            self.embed.data[ind][mask] = sampled
            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, "h ... d -> h (...) d")
        self.replace(batch_samples, batch_mask=expired_codes)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x, sample_codebook_temp=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        flatten, ps = pack_one(x, "h * d")
        if freeze_codebook and not bool(self.initted.item()):
            raise RuntimeError(
                "Cannot use an uninitialised codebook with freeze_codebook=True. "
                "Run at least one training initialisation pass or load a checkpoint."
            )
        self.init_embed_(flatten)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        # Euclidean distance  (negative → closer = higher score)
        x2 = reduce(flatten**2, "h n d -> h n", "sum")
        e2 = reduce(embed**2, "h c d -> h c", "sum")
        xy = einsum("h n d, h c d -> h n c", flatten, embed) * -2
        dist = -(rearrange(x2, "h n -> h n 1") + rearrange(e2, "h c -> h 1 c") + xy).clamp(min=1e-12).sqrt()

        embed_ind, embed_onehot = gumbel_sample(
            dist,
            dim=-1,
            temperature=sample_codebook_temp,
            training=self.training,
        )
        embed_ind = unpack_one(embed_ind, ps, "h *")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            quantize = einsum("h b n c, h c d -> h b n d", unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        # EMA codebook update
        if self.training and self.ema_update and not freeze_codebook:
            if self._ema_step_counter < self.ema_warmup_steps:
                self._ema_step_counter += 1
            else:
                cluster_size = embed_onehot.sum(dim=1)
                ema_inplace(self.cluster_size.data, cluster_size, self.decay)

                embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
                ema_inplace(self.embed_avg.data, embed_sum, self.decay)

                cluster_size = laplace_smoothing(
                    self.cluster_size, self.codebook_size, self.eps
                ) * self.cluster_size.sum(dim=-1, keepdim=True)
                embed_normalized = self.embed_avg / rearrange(cluster_size, "... -> ... 1")
                self.embed.data.copy_(embed_normalized)
                self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = (rearrange(t, "1 ... -> ...") for t in (quantize, embed_ind))

        dist = unpack_one(dist, ps, "h * d")

        return quantize, embed_ind, dist


# ===========================================================================
# VectorQuantize — top-level wrapper with projection + commitment loss
# ===========================================================================


class VectorQuantize(nn.Module):
    """Vector Quantization layer wrapping EuclideanCodebook.

    Adds input/output projection, commitment loss, and codebook accessors.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        decay: float = 0.8,
        eps: float = 1e-5,
        ema_warmup_steps: int = 0,
        freeze_codebook: bool = False,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 0,
        commitment_weight: float = 1.0,
        sample_codebook_temp: float = 1.0,
        ema_update: bool = True,
        learnable_codebook: bool = False,
    ):
        super().__init__()
        self.dim = dim

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight

        self.learnable_codebook = learnable_codebook
        assert not (ema_update and learnable_codebook), "learnable codebook not compatible with EMA update"

        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            num_codebooks=1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            ema_warmup_steps=ema_warmup_steps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            learnable_codebook=learnable_codebook,
            sample_codebook_temp=sample_codebook_temp,
            ema_update=ema_update,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        """Return codebook embedding (codebook_size, dim)."""
        return rearrange(self._codebook.embed, "1 c d -> c d")

    @codebook.setter
    def codebook(self, codes):
        self._codebook.embed.copy_(rearrange(codes, "c d -> 1 c d"))

    def get_codes_from_indices(self, indices):
        """Look up codebook vectors from indices (B,)."""
        codes = self.codebook[indices]  # (B, dim)
        return codes

    def forward(self, x, indices=None, sample_codebook_temp=None, freeze_codebook=False):
        """
        Args:
            x:  (B, D)       single vector per sample
                (B, N, D)   sequence of vectors
            indices:  if given, return CE loss instead of commitment loss
        Returns:
            quantize:  (B, D) or (B, N, D)
            embed_ind: (B,) or (B, N)  codebook indices
            loss:      scalar commitment loss (or CE loss if indices given)
        """
        only_one = x.ndim == 2
        if only_one:
            x = rearrange(x, "b d -> b 1 d")

        return_loss = exists(indices)

        # project in
        x = self.project_in(x)  # (B, N, codebook_dim)
        x = rearrange(x, "b n d -> 1 b n d")  # (1, B, N, codebook_dim)

        # forward through Euclidean codebook
        quantize, embed_ind, distances = self._codebook(
            x,
            sample_codebook_temp=sample_codebook_temp,
            freeze_codebook=freeze_codebook,
        )

        embed_ind = rearrange(embed_ind, "1 b n -> b n")  # (B, N)

        # Commitment target is available in both train and eval modes so
        # validation reports a real commitment metric.  Straight-through is
        # only needed while training.
        commit_quantize = quantize.detach()
        if self.training:
            quantize = x + (quantize - x).detach()  # straight-through

        # project out
        quantize = rearrange(quantize, "1 b n d -> b n d")
        quantize = self.project_out(quantize)  # (B, N, dim)

        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")
            embed_ind = rearrange(embed_ind, "b 1 -> b")

        # loss
        loss = x.new_zeros((1,))

        if return_loss:
            ce_loss = F.cross_entropy(
                rearrange(distances, "1 b n l -> b l n"),
                indices,
                ignore_index=-1,
            )
            return quantize, ce_loss

        if self.commitment_weight > 0:
            # In eval mode callers normally wrap the forward pass in
            # torch.no_grad(), so this remains a metric-only computation.
            commit_loss = F.mse_loss(commit_quantize, x)
            loss = loss + commit_loss * self.commitment_weight

        return quantize, embed_ind, loss
