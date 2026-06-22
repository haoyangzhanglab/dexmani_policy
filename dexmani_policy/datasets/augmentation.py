import numpy as np
import torch
import torchvision.transforms.v2 as v2

_BT601_LUMA = (0.299, 0.587, 0.114)
"""BT.601 luma coefficients for RGB-to-grayscale conversion."""

class Aug:
    """Prob-gated augmentation base.

    Subclasses implement ``_augment(x)`` which **must** modify *x*
    in-place — the caller guarantees that *x* is a detached copy
    when ``_augment`` runs (via ``apply_augmentation`` in the dataset).

    Prob-gating and defensive copying are handled by
    :meth:`~dexmani_policy.datasets.base_dataset.BaseDataset.apply_augmentation`,
    which iterates over augmentors, decides per-augmentor whether to trigger,
    and makes a single ``.copy()`` before the first trigger.
    """

    __slots__ = ('prob',)

    def __init__(self, prob=1.0):
        self.prob = prob

    def _augment(self, x):
        """Modify *x* **in-place**.  Caller guarantees *x* is a detached copy."""
        raise NotImplementedError

class PointColorJitter(Aug):
    """HSV color jitter for point cloud RGB channels (last 3 dims).

    Reference: ManiFlow (CoRL 2025). Input ``(T,N,C) float32, C>=6``.
    Modifies the RGB channels **in-place** on a pre-copied array.
    """

    __slots__ = ('brightness', 'contrast', 'saturation', 'hue')

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05,
                 prob=1.0):
        super().__init__(prob=prob)
        self.brightness = self._make_range(brightness, center=1.0)
        self.contrast   = self._make_range(contrast,   center=1.0)
        self.saturation = self._make_range(saturation, center=1.0)
        self.hue        = self._make_range(hue,        center=0.0, symmetric=True)

    @staticmethod
    def _make_range(value, center, symmetric=False):
        if symmetric:
            if isinstance(value, (int, float)):
                return [-abs(value), abs(value)]
            return [value[0], value[1]]
        if isinstance(value, (int, float)):
            value = abs(value)
            return [max(0.0, center - value), center + value]
        return [value[0], value[1]]

    # ---------- in-place augment ----------
    def _augment(self, x):
        if x.shape[-1] < 6:
            return

        # Only touch the RGB channels — (T,N,C) → flat (−1,3) for batch HSV.
        rgb = x[..., -3:].reshape(-1, 3)

        self._apply_brightness_ip(rgb)
        self._apply_contrast_ip(rgb)
        self._apply_saturation_ip(rgb)
        self._apply_hue_ip(rgb)
        np.clip(rgb, 0.0, 1.0, out=rgb)

    # ---------- in-place colour helpers ----------
    def _apply_brightness_ip(self, rgb):
        if self.brightness[0] == self.brightness[1]:
            return
        rgb *= np.random.uniform(*self.brightness)

    def _apply_contrast_ip(self, rgb):
        if self.contrast[0] == self.contrast[1]:
            return
        factor = np.random.uniform(*self.contrast)
        rgb -= 0.5
        rgb *= factor
        rgb += 0.5

    def _apply_saturation_ip(self, rgb):
        if self.saturation[0] == self.saturation[1]:
            return
        factor = np.random.uniform(*self.saturation)
        # gray is (N,) — no [:, None] needed since we add channel-wise
        gray = np.dot(rgb, _BT601_LUMA)  # (N,)
        inv_factor = 1.0 - factor
        rgb *= factor
        rgb[:, 0] += gray * inv_factor
        rgb[:, 1] += gray * inv_factor
        rgb[:, 2] += gray * inv_factor

    def _apply_hue_ip(self, rgb):
        if self.hue[0] == self.hue[1]:
            return
        shift = np.random.uniform(*self.hue)
        if abs(shift) < 1e-6:
            return
        hsv = self._rgb_to_hsv(rgb)
        hsv[:, 0] = (hsv[:, 0] + shift) % 1.0
        rgb[...] = self._hsv_to_rgb(hsv)

    @staticmethod
    def _rgb_to_hsv(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        max_v, min_v = np.max(rgb, axis=-1), np.min(rgb, axis=-1)
        diff = max_v - min_v
        max_idx = np.argmax(rgb, axis=-1)

        h = np.zeros_like(max_v)
        mask = diff != 0

        rm = mask & (max_idx == 0)
        h[rm] = ((g[rm] - b[rm]) / diff[rm]) / 6.0
        gm = mask & (max_idx == 1)
        h[gm] = ((b[gm] - r[gm]) / diff[gm] + 2.0) / 6.0
        bm = mask & (max_idx == 2)
        h[bm] = ((r[bm] - g[bm]) / diff[bm] + 4.0) / 6.0

        h %= 1.0
        s = np.where(max_v != 0, diff / max_v, 0.0)
        return np.stack([h, s, max_v], axis=-1)

    @staticmethod
    def _hsv_to_rgb(hsv):
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        h6 = h * 6.0
        i = np.floor(h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        rgb = np.zeros_like(hsv)
        for idx, (r_src, g_src, b_src) in enumerate([
            (v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q),
        ]):
            mask = i == idx
            rgb[mask, 0] = r_src[mask]
            rgb[mask, 1] = g_src[mask]
            rgb[mask, 2] = b_src[mask]
        return rgb

class PointDropout(Aug):
    """Random point dropout with replacement — simulates sparse or partial scans.

    Dropped points are replaced by randomly sampling from the *kept* points,
    creating natural duplicate density variations that are harder for the
    encoder to exploit than zero-filling or single-anchor replacement (R3D, 2026).

    Each T frame gets an independent dropout mask.  Modifies the array
    **in-place** on a pre-copied input.
    """

    __slots__ = ('dropout_ratio',)

    def __init__(self, dropout_ratio=0.3, prob=1.0):
        super().__init__(prob=prob)
        self.dropout_ratio = dropout_ratio

    def _augment(self, x):
        # x: (T, N, C) — guaranteed to be a detached copy
        T, N = x.shape[:2]
        if N <= 1:
            return  # can't dropout from a point cloud with ≤ 1 point
        n_drop = max(1, min(int(N * self.dropout_ratio), N - 1))

        # Boolean mask instead of np.setdiff1d (O(N) vs O(N log N)).
        all_idx = np.arange(N)
        for t in range(T):
            drop_idx = np.random.choice(all_idx, size=n_drop, replace=False)

            keep_mask = np.ones(N, dtype=bool)
            keep_mask[drop_idx] = False
            keep_idx = all_idx[keep_mask]       # O(N) — no sort

            fill_idx = np.random.choice(keep_idx, size=n_drop, replace=True)
            x[t, drop_idx] = x[t, fill_idx]

class PointCoordNoiseAug(Aug):
    """Clipped Gaussian noise on a random subset of point XYZ coordinates.

    Unlike ``StateNoiseAug`` which perturbs proprioceptive state, this adds
    per-point geometric noise to simulate real sensor measurement error.
    Noise is applied only to a random subset of points each call, preventing
    the model from memorising exact point positions (R3D, 2026).

    The noise is clipped to ±clip_range (default 2σ) to guard against extreme
    outliers that would push normalised coordinates outside [-1, 1].

    Modifies the first 3 channels **in-place** on a pre-copied input.
    """

    __slots__ = ('noise_std', 'ratio', 'clip_range')

    def __init__(self, noise_std=0.002, ratio=0.3, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.ratio = float(ratio)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def _augment(self, x):
        # x: (T, N, C) — noise only on the first 3 channels (XYZ)
        if self.noise_std <= 0 or self.ratio <= 0:
            return

        T, N = x.shape[:2]
        n_noisy = max(1, int(N * self.ratio))

        # Per-frame random subset — different points each frame
        dtype = x.dtype
        for t in range(T):
            idx = np.random.choice(N, size=n_noisy, replace=False)
            noise = np.random.normal(0, self.noise_std, (n_noisy, 3))
            noise = np.clip(noise, -self.clip_range, self.clip_range)
            x[t, idx, :3] += noise.astype(dtype)

class PointColorNoiseAug(Aug):
    """Per-channel independent Gaussian noise on point cloud RGB.

    Models camera sensor noise (per-pixel, per-channel), complementary to
    ``PointColorJitter`` which models scene lighting changes (global HSV
    transform).  Both are applied in R3D-Policy (2026) as independent
    augmentation stages.

    Noise is clipped to ``±clip_range`` (default 2σ) and the result is
    clamped to [0, 1] to keep RGB values valid.

    Modifies the last 3 channels **in-place** on a pre-copied input.
    Requires C >= 6 (xyz + rgb); silently returns for coord-only point clouds.
    """

    __slots__ = ('noise_std', 'clip_range')

    def __init__(self, noise_std=0.01, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def _augment(self, x):
        if x.shape[-1] < 6 or self.noise_std <= 0:
            return
        rgb = x[..., -3:]
        noise = np.random.normal(0, self.noise_std, rgb.shape)
        noise = np.clip(noise, -self.clip_range, self.clip_range)
        rgb += noise.astype(x.dtype)
        np.clip(rgb, 0.0, 1.0, out=rgb)

class StateNoiseAug(Aug):
    """Clipped Gaussian noise on proprioceptive state.

    Noise is clipped to ``±clip_range`` (default 2σ) to guard against extreme
    outliers that would push normalised state outside [-1, 1].

    Reference: R3D-Policy (2026) ``add_noise()`` — all modalities
    (xyz/rgb/agent_pos) use ``clip_range = 2 * noise_std``.

    NOTE: This is data augmentation — it generates plausible sensor readings
    by adding noise. It is NOT the same as modality dropout (``modality_dropout_probs``
    in agent config), which zeros out the entire modality to force multi-modal
    robustness.

    Modifies the array **in-place** on a pre-copied input.
    """

    __slots__ = ('noise_std', 'clip_range')

    def __init__(self, noise_std=0.005, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def _augment(self, x):
        noise = np.random.normal(0, self.noise_std, x.shape)
        noise = np.clip(noise, -self.clip_range, self.clip_range)
        x += noise.astype(x.dtype)

class ImageAug:
    """Color + blur + noise for pre-resized float32 CHW tensors (no numpy roundtrip).

    Reference: DexUMI (brightness=0.6 for challenging lighting).
    Each sub-augmentation is independently probabilistic.

    Performance note: all probabilistic gates are decided from a single
    ``torch.rand`` call instead of one per sub-transform, reducing CUDA
    kernel launches.
    """

    __slots__ = ('prob', 'color_jitter', 'grayscale', 'noise', 'noise_prob',
                 'blur', 'blur_prob')

    def __init__(
        self,
        brightness=0.6,
        contrast=0.3,
        saturation=0.3,
        hue=0.08,
        grayscale_prob=0.2,
        noise_std=0.0,
        noise_prob=1.0,
        blur_kernel=3,
        blur_sigma=(0.1, 2.0),
        blur_prob=0.2,
        prob=1.0,
    ):
        self.prob = prob

        self.color_jitter = v2.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )
        self.grayscale = v2.RandomGrayscale(p=grayscale_prob) if grayscale_prob > 0 else None

        self.noise = v2.GaussianNoise(mean=0.0, sigma=noise_std) if noise_std > 0 else None
        self.noise_prob = noise_prob

        self.blur = v2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma) if blur_kernel > 0 else None
        self.blur_prob = blur_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, 3, H, W) float32 [0, 1] → same shape."""
        # Fuse prob gate + noise + blur decisions. When prob=1.0 the prob gate
        # is skipped and only noise/blur decisions are generated (save 1 RNG value).
        need_prob = self.prob < 1.0
        n_rand = 3 if need_prob else 2
        rand_vals = torch.rand(n_rand, device=x.device)

        if need_prob and rand_vals[0].item() > self.prob:
            return x

        noise_idx = 1 if need_prob else 0
        blur_idx = 2 if need_prob else 1

        x = self.color_jitter(x)

        if self.grayscale is not None:
            x = self.grayscale(x)

        if self.noise is not None and rand_vals[noise_idx].item() <= self.noise_prob:
            x = self.noise(x)

        if self.blur is not None and rand_vals[blur_idx].item() <= self.blur_prob:
            x = self.blur(x)

        return x.clamp(0, 1)
