import numpy as np
import torch
import torchvision.transforms.v2 as v2

_BT601_LUMA = (0.299, 0.587, 0.114)
"""BT.601 luma coefficients for RGB-to-grayscale conversion."""



class Aug:
    """Prob-gated augmentation base. Subclasses implement ``_augment(x)``."""

    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, x):
        if np.random.random() > self.prob:
            return x
        return self._augment(x)

    def _augment(self, x):
        raise NotImplementedError



class PointColorJitter(Aug):
    """HSV color jitter for point cloud RGB channels (last 3 dims).

    Reference: ManiFlow (CoRL 2025). Input ``(T,N,C) float32, C>=6``.
    """

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

    def _augment(self, x):
        if x.shape[-1] < 6:
            return x

        pc = x.copy()
        rgb = pc[..., -3:]
        orig_shape = rgb.shape
        rgb = rgb.reshape(-1, 3)

        rgb = self._apply_brightness(rgb)
        rgb = self._apply_contrast(rgb)
        rgb = self._apply_saturation(rgb)
        rgb = self._apply_hue(rgb)
        rgb = np.clip(rgb, 0.0, 1.0)

        pc[..., -3:] = rgb.reshape(orig_shape)
        return pc

    def _apply_brightness(self, rgb):
        if self.brightness[0] == self.brightness[1]:
            return rgb
        return rgb * np.random.uniform(*self.brightness)

    def _apply_contrast(self, rgb):
        if self.contrast[0] == self.contrast[1]:
            return rgb
        factor = np.random.uniform(*self.contrast)
        return (rgb - 0.5) * factor + 0.5

    def _apply_saturation(self, rgb):
        if self.saturation[0] == self.saturation[1]:
            return rgb
        factor = np.random.uniform(*self.saturation)
        gray = np.dot(rgb, _BT601_LUMA)[:, None]
        return rgb * factor + gray * (1.0 - factor)

    def _apply_hue(self, rgb):
        if self.hue[0] == self.hue[1]:
            return rgb
        shift = np.random.uniform(*self.hue)
        if abs(shift) < 1e-6:
            return rgb
        hsv = self._rgb_to_hsv(rgb)
        hsv[:, 0] = (hsv[:, 0] + shift) % 1.0
        return self._hsv_to_rgb(hsv)

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

    Each T frame gets an independent dropout mask.
    """

    def __init__(self, dropout_ratio=0.3, prob=1.0):
        super().__init__(prob=prob)
        self.dropout_ratio = dropout_ratio

    def _augment(self, x):
        # x: (T, N, C)
        T, N = x.shape[:2]
        n_drop = max(1, int(N * self.dropout_ratio))
        n_keep = N - n_drop

        pc = x.copy()
        for t in range(T):
            all_idx = np.arange(N)
            drop_idx = np.random.choice(all_idx, size=n_drop, replace=False)

            keep_idx = np.setdiff1d(all_idx, drop_idx)
            fill_idx = np.random.choice(keep_idx, size=n_drop, replace=True)

            pc[t, drop_idx] = pc[t, fill_idx]
        return pc


class PointCoordNoiseAug(Aug):
    """Clipped Gaussian noise on a random subset of point XYZ coordinates.

    Unlike ``StateNoiseAug`` which perturbs proprioceptive state, this adds
    per-point geometric noise to simulate real sensor measurement error.
    Noise is applied only to a random subset of points each call, preventing
    the model from memorising exact point positions (R3D, 2026).

    The noise is clipped to ±clip_range (default 2σ) to guard against extreme
    outliers that would push normalised coordinates outside [-1, 1].
    """

    def __init__(self, noise_std=0.002, ratio=0.3, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.ratio = float(ratio)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def _augment(self, x):
        # x: (T, N, C) — noise only on the first 3 channels (XYZ)
        if self.noise_std <= 0 or self.ratio <= 0:
            return x

        T, N = x.shape[:2]
        n_noisy = max(1, int(N * self.ratio))

        # Per-frame random subset — different points each frame
        pc = x.copy()
        for t in range(T):
            idx = np.random.choice(N, size=n_noisy, replace=False)
            noise = np.random.normal(0, self.noise_std, (n_noisy, 3))
            noise = np.clip(noise, -self.clip_range, self.clip_range)
            pc[t, idx, :3] += noise.astype(pc.dtype)
        return pc


class StateNoiseAug(Aug):
    """Small Gaussian noise on proprioceptive state.

    NOTE: This is data augmentation — it generates plausible sensor readings
    by adding noise. It is NOT the same as modality dropout (``modality_dropout_probs``
    in agent config), which zeros out the entire modality to force multi-modal
    robustness. See ``configs/augmentation_example.yaml`` for the full distinction.
    """

    def __init__(self, noise_std=0.005, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = noise_std

    def _augment(self, x):
        return x + np.random.normal(0, self.noise_std, x.shape).astype(x.dtype)



class ImageAug:
    """Color + blur + noise for pre-resized float32 CHW tensors (no numpy roundtrip).

    Reference: DexUMI (brightness=0.6 for challenging lighting).
    Each sub-augmentation is independently probabilistic.
    """

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
        if self.prob < 1.0 and torch.rand(1, device=x.device).item() > self.prob:
            return x

        x = self.color_jitter(x)

        if self.grayscale is not None:
            x = self.grayscale(x)

        if self.noise is not None and torch.rand(1, device=x.device).item() <= self.noise_prob:
            x = self.noise(x)

        if self.blur is not None and torch.rand(1, device=x.device).item() <= self.blur_prob:
            x = self.blur(x)

        return x.clamp(0, 1)
