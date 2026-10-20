import numpy as np
import torch
import torchvision.transforms.v2 as v2

from dexmani_policy.utils.config import positive_int

BT601_LUMA = (0.299, 0.587, 0.114)


def validate_random_keep_points(keep_points, *, num_points=None, num_patches=None, group_size=None):
    """Validate optional dataset subsampling and the receiving geometry sizes."""
    if keep_points is None:
        return
    positive_int(keep_points, "random_keep_points")
    if num_points is not None and num_points < keep_points:
        raise ValueError(f"point_cloud N={num_points} must be >= random_keep_points={keep_points}")
    for name, size in (("num_patches", num_patches), ("group_size", group_size)):
        if size is not None and keep_points < size:
            raise ValueError(f"random_keep_points={keep_points} must be >= {name}={size}")


def random_sample_points(pointcloud: np.ndarray, keep_points: int) -> np.ndarray:
    """Sample each frame without replacement, using shared indices for all channels."""
    t, n, _ = pointcloud.shape
    scores = np.random.random((t, n))
    indices = np.argpartition(scores, kth=keep_points - 1, axis=1)[:, :keep_points]
    return np.ascontiguousarray(np.take_along_axis(pointcloud, indices[..., None], axis=1))


def validate_pc_color_augmentation(pc_dim, augmentation_cfg):
    """Require XYZRGB channels for enabled point color transforms."""
    if pc_dim is None or pc_dim >= 6:
        return
    pc = (augmentation_cfg or {}).get("pc") or {}
    for key in ("color", "color_noise"):
        config = pc.get(key)
        if config and config.get("prob", 1) > 0:
            raise ValueError(
                f"XYZ-only or incomplete RGB point clouds cannot use color augmentation ({key}); "
                "set prob=0 or use XYZRGB channels"
            )


class Augmentation:
    """In-place transform; the dataset applies probability gating to owned sample arrays."""

    __slots__ = ("prob",)

    def __init__(self, prob=1.0):
        self.prob = prob

    def augment(self, x):
        raise NotImplementedError


class PointColorJitter(Augmentation):
    """Additive brightness and shared contrast, saturation and hue on the last RGB channels."""

    __slots__ = ("brightness", "contrast", "saturation", "hue")

    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.08, prob=1.0):
        super().__init__(prob=prob)
        self.brightness = self.make_range(brightness, center=0.0, symmetric=True)
        self.contrast = self.make_range(contrast, center=1.0)
        self.saturation = self.make_range(saturation, center=1.0)
        self.hue = self.make_range(hue, center=0.0, symmetric=True)

    @staticmethod
    def make_range(value, center, symmetric=False):
        if symmetric:
            if isinstance(value, (int, float)):
                return [-abs(value), abs(value)]
            return [value[0], value[1]]
        if isinstance(value, (int, float)):
            value = abs(value)
            return [max(0.0, center - value), center + value]
        return [value[0], value[1]]

    def augment(self, x):
        rgb = x[..., -3:].reshape(-1, 3)

        self.apply_brightness(rgb)
        self.apply_contrast(rgb)
        self.apply_saturation(rgb)

        # Additive brightness can leave the [0, 1] domain required by HSV.
        np.clip(rgb, 0.0, 1.0, out=rgb)
        self.apply_hue(rgb)
        np.clip(rgb, 0.0, 1.0, out=rgb)

    def apply_brightness(self, rgb):
        if self.brightness[0] == self.brightness[1]:
            delta = self.brightness[0]
        else:
            delta = np.random.uniform(*self.brightness)
        if delta == 0:
            return
        rgb += delta

    def apply_contrast(self, rgb):
        if self.contrast[0] == self.contrast[1]:
            factor = self.contrast[0]
        else:
            factor = np.random.uniform(*self.contrast)
        if factor == 1:
            return
        mean = rgb.mean()
        rgb -= mean
        rgb *= factor
        rgb += mean

    def apply_saturation(self, rgb):
        if self.saturation[0] == self.saturation[1]:
            factor = self.saturation[0]
        else:
            factor = np.random.uniform(*self.saturation)
        if factor == 1:
            return
        gray = np.dot(rgb, BT601_LUMA)
        inv_factor = 1.0 - factor
        rgb *= factor
        rgb[:, 0] += gray * inv_factor
        rgb[:, 1] += gray * inv_factor
        rgb[:, 2] += gray * inv_factor

    def apply_hue(self, rgb):
        if self.hue[0] == self.hue[1]:
            shift = self.hue[0]
        else:
            shift = np.random.uniform(*self.hue)
        if abs(shift) < 1e-6:
            return
        hsv = self.rgb_to_hsv(rgb)
        hsv[:, 0] = (hsv[:, 0] + shift) % 1.0
        rgb[...] = self.hsv_to_rgb(hsv)

    @staticmethod
    def rgb_to_hsv(rgb):
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
        # np.where evaluates both branches, so it does not prevent 0 / 0.
        s = np.zeros_like(diff)
        np.divide(diff, max_v, out=s, where=max_v != 0)
        return np.stack([h, s, max_v], axis=-1)

    @staticmethod
    def hsv_to_rgb(hsv):
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        h6 = h * 6.0
        i = np.floor(h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        rgb = np.zeros_like(hsv)
        for idx, (r_src, g_src, b_src) in enumerate(
            [
                (v, t, p),
                (q, v, p),
                (p, v, t),
                (p, q, v),
                (t, p, v),
                (v, p, q),
            ]
        ):
            mask = i == idx
            rgb[mask, 0] = r_src[mask]
            rgb[mask, 1] = g_src[mask]
            rgb[mask, 2] = b_src[mask]
        return rgb


class PointDropout(Augmentation):
    """Replace dropped rows with the first point, sampling a dropout ratio per frame."""

    __slots__ = ("max_dropout_ratio",)

    def __init__(self, max_dropout_ratio=0.8, prob=1.0):
        if not 0 <= max_dropout_ratio <= 1:
            raise ValueError("max_dropout_ratio must be between 0 and 1")
        super().__init__(prob=prob)
        self.max_dropout_ratio = float(max_dropout_ratio)

    def augment(self, x):
        if self.max_dropout_ratio <= 0:
            return

        T, N = x.shape[:2]
        if N == 0:
            return

        for t in range(T):
            ratio = np.random.uniform(0.0, self.max_dropout_ratio)
            drop_mask = np.random.random(N) <= ratio
            if np.any(drop_mask):
                anchor = x[t, 0].copy()
                x[t, drop_mask] = anchor


class PointCoordinateNoise(Augmentation):
    """Clipped Gaussian noise on all XYZ coordinates, in raw coordinate units."""

    __slots__ = ("noise_std", "clip_range")

    def __init__(self, noise_std=0.002, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def augment(self, x):
        if self.noise_std <= 0:
            return

        T, N = x.shape[:2]
        dtype = x.dtype
        for t in range(T):
            noise = np.random.normal(0, self.noise_std, (N, 3))
            noise = np.clip(noise, -self.clip_range, self.clip_range)
            x[t, :, :3] += noise.astype(dtype)


class PointColorNoise(Augmentation):
    """Clipped Gaussian noise on the last RGB channels, clamped to [0, 1]."""

    __slots__ = ("noise_std", "clip_range")

    def __init__(self, noise_std=0.01, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def augment(self, x):
        if self.noise_std <= 0:
            return
        rgb = x[..., -3:]
        noise = np.random.normal(0, self.noise_std, rgb.shape)
        noise = np.clip(noise, -self.clip_range, self.clip_range)
        rgb += noise.astype(x.dtype)
        np.clip(rgb, 0.0, 1.0, out=rgb)


class StateNoise(Augmentation):
    """Clipped Gaussian noise on proprioception, in raw state units."""

    __slots__ = ("noise_std", "clip_range")

    def __init__(self, noise_std=0.0002, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def augment(self, x):
        noise = np.random.normal(0, self.noise_std, x.shape)
        noise = np.clip(noise, -self.clip_range, self.clip_range)
        x += noise.astype(x.dtype)


class ImageAugmentation:
    """Color jitter, grayscale, pixel noise and blur on float32 observation windows."""

    __slots__ = ("prob", "color_jitter", "grayscale", "noise", "noise_prob", "blur", "blur_prob")

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
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.grayscale = v2.RandomGrayscale(p=grayscale_prob) if grayscale_prob > 0 else None

        self.noise = v2.GaussianNoise(mean=0.0, sigma=noise_std) if noise_std > 0 else None
        self.noise_prob = noise_prob

        self.blur = (
            v2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma) if blur_kernel > 0 else None
        )
        self.blur_prob = blur_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, 3, H, W) float32 [0, 1] → same shape."""
        # Preserve RNG consumption: prob=1 omits the outer probability draw.
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


# NumPy transforms run in this order before point subsampling and RGB processing.
AUGMENTATION_SPECS = (
    ("pc", PointCoordinateNoise, "coord_noise", "point_cloud"),
    ("pc", PointColorJitter, "color", "point_cloud"),
    ("pc", PointColorNoise, "color_noise", "point_cloud"),
    ("pc", PointDropout, "dropout", "point_cloud"),
    ("state", StateNoise, "noise", "joint_state"),
)
