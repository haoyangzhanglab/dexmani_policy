import numpy as np
import torch

BT601_LUMA = (0.299, 0.587, 0.114)


def clipped_gaussian_noise(shape, std, clip_range, dtype):
    noise = np.random.normal(0, std, shape)
    np.clip(noise, -clip_range, clip_range, out=noise)
    return noise.astype(dtype, copy=False)


def sample_range(bounds):
    return bounds[0] if bounds[0] == bounds[1] else np.random.uniform(*bounds)


def random_sample_points(pointcloud: np.ndarray, keep_points: int) -> np.ndarray:
    """Sample each frame without replacement, using shared indices for all channels."""
    t, n, _ = pointcloud.shape
    scores = np.random.random((t, n))
    indices = np.argpartition(scores, kth=keep_points - 1, axis=1)[:, :keep_points]
    return np.ascontiguousarray(np.take_along_axis(pointcloud, indices[..., None], axis=1))


class Augmentation:
    """In-place transform; the dataset applies probability gating to owned sample arrays."""

    def __init__(self, prob=1.0):
        self.prob = prob

    def augment(self, x):
        raise NotImplementedError


class PointColorJitter(Augmentation):
    """Additive brightness and shared contrast, saturation and hue on the last RGB channels."""

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
        delta = sample_range(self.brightness)
        if delta == 0:
            return
        rgb += delta

    def apply_contrast(self, rgb):
        factor = sample_range(self.contrast)
        if factor == 1:
            return
        mean = rgb.mean()
        rgb -= mean
        rgb *= factor
        rgb += mean

    def apply_saturation(self, rgb):
        factor = sample_range(self.saturation)
        if factor == 1:
            return
        gray = np.dot(rgb, BT601_LUMA)
        inv_factor = 1.0 - factor
        rgb *= factor
        rgb[:, 0] += gray * inv_factor
        rgb[:, 1] += gray * inv_factor
        rgb[:, 2] += gray * inv_factor

    def apply_hue(self, rgb):
        shift = sample_range(self.hue)
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

    def __init__(self, noise_std=0.002, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def augment(self, x):
        if self.noise_std <= 0:
            return

        xyz = x[..., :3]
        xyz += clipped_gaussian_noise(xyz.shape, self.noise_std, self.clip_range, x.dtype)


class PointColorNoise(Augmentation):
    """Clipped Gaussian noise on the last RGB channels, clamped to [0, 1]."""

    def __init__(self, noise_std=0.01, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def augment(self, x):
        if self.noise_std <= 0:
            return
        rgb = x[..., -3:]
        rgb += clipped_gaussian_noise(rgb.shape, self.noise_std, self.clip_range, x.dtype)
        np.clip(rgb, 0.0, 1.0, out=rgb)


class StateNoise(Augmentation):
    """Noise is applied in raw state units."""

    def __init__(self, noise_std=0.0002, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std

    def augment(self, x):
        x += clipped_gaussian_noise(x.shape, self.noise_std, self.clip_range, x.dtype)


class ImageAugmentation:
    """Color jitter, grayscale, pixel noise and blur on float32 observation windows."""

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
        import torchvision.transforms.v2 as v2

        operations = []
        if prob > 0:
            if any((brightness, contrast, saturation, hue)):
                operations.append(v2.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                ))
            if grayscale_prob > 0:
                operations.append(v2.RandomGrayscale(p=grayscale_prob))
            if noise_std > 0 and noise_prob > 0:
                operations.append(v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=noise_std)], p=noise_prob))
            if blur_kernel > 0 and blur_prob > 0:
                operations.append(v2.RandomApply([
                    v2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma),
                ], p=blur_prob))
        self.transform = v2.Compose(operations) if operations else torch.nn.Identity()
        if operations and 0 < prob < 1:
            self.transform = v2.RandomApply([self.transform], p=prob)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Transform decisions and color parameters are shared across the T-frame window.
        return self.transform(x).clamp(0, 1)


# NumPy transforms run in this order before point subsampling and RGB processing.
AUGMENTATION_SPECS = (
    ("pc", PointCoordinateNoise, "coord_noise", "point_cloud"),
    ("pc", PointColorJitter, "color", "point_cloud"),
    ("pc", PointColorNoise, "color_noise", "point_cloud"),
    ("pc", PointDropout, "dropout", "point_cloud"),
    ("state", StateNoise, "noise", "joint_state"),
)
