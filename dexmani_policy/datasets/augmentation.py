import numpy as np
import torch
import torchvision.transforms.v2 as v2



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
        gray = np.dot(rgb, [0.299, 0.587, 0.114])[:, None]
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


class PointSpatialAug(Aug):
    """Z-axis rotation + XY translation + uniform scale. Same transform across all T frames.

    Reference: RISE (IROS 2024). Input ``(T,N,C) float32``, XYZ in ``[...,:3]``.
    """

    def __init__(self, rot_z=15.0, trans_xy=0.10, scale=1.02, prob=1.0):
        super().__init__(prob=prob)
        self.rot_z = float(rot_z)
        self.trans_xy = float(trans_xy)
        self.scale = float(scale)

    def _augment(self, x):
        pc = x.copy()
        xyz = pc[..., :3]

        angle = np.deg2rad(np.random.uniform(-self.rot_z, self.rot_z))
        dx = np.random.uniform(-self.trans_xy, self.trans_xy)
        dy = np.random.uniform(-self.trans_xy, self.trans_xy)
        s = np.random.uniform(min(1.0 / self.scale, self.scale),
                              max(1.0 / self.scale, self.scale))

        cos_a, sin_a = np.cos(angle), np.sin(angle)

        rx = xyz[..., 0] * cos_a - xyz[..., 1] * sin_a
        ry = xyz[..., 0] * sin_a + xyz[..., 1] * cos_a
        pc[..., 0] = rx * s + dx
        pc[..., 1] = ry * s + dy
        pc[..., 2] = xyz[..., 2] * s

        # rotate normals (channels 3:6) if present — same rot, no translation/scale
        if pc.shape[-1] >= 6:
            nxyz = x[..., 3:6]
            pc[..., 3] = nxyz[..., 0] * cos_a - nxyz[..., 1] * sin_a
            pc[..., 4] = nxyz[..., 0] * sin_a + nxyz[..., 1] * cos_a

        return pc


class PointDropout(Aug):
    """Random point dropout — simulates occlusion in dexterous manipulation.
    Dropped points are zeroed. Same mask across all T frames.
    """

    def __init__(self, dropout_ratio=0.1, prob=1.0):
        super().__init__(prob=prob)
        self.dropout_ratio = dropout_ratio

    def _augment(self, x):
        k = int(x.shape[1] * self.dropout_ratio)
        if k == 0:
            return x
        drop_idx = np.random.choice(x.shape[1], size=k, replace=False)
        pc = x.copy()
        pc[:, drop_idx] = 0.0
        return pc



class StateNoiseAug(Aug):
    """Small Gaussian noise on proprioceptive state."""

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

        return x.clamp_(0, 1)
