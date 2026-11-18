import numpy as np
import torch

BT601_LUMA = (0.299, 0.587, 0.114)


def clipped_gaussian_noise(shape, std, clip_range, dtype):
    noise = np.random.normal(0, std, shape)
    np.clip(noise, -clip_range, clip_range, out=noise)
    return noise.astype(dtype, copy=False)


def sample_range(bounds):
    return bounds[0] if bounds[0] == bounds[1] else np.random.uniform(*bounds)


class Augmentation:
    """In-place transform; the dataset applies probability gating to owned sample arrays."""

    def __init__(self, prob=1.0):
        if not np.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("augmentation probability must be finite in [0,1]")
        self.prob = prob

    def augment(self, x):
        raise NotImplementedError


class PointColorJitter(Augmentation):
    """Additive brightness and shared contrast and saturation on the last RGB channels."""

    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, prob=1.0):
        super().__init__(prob=prob)
        self.brightness = self.make_range(brightness, center=0.0, symmetric=True)
        self.contrast = self.make_range(contrast, center=1.0)
        self.saturation = self.make_range(saturation, center=1.0)

    @staticmethod
    def make_range(value, center, symmetric=False):
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("color strengths must be finite non-negative scalars")
        return [-value, value] if symmetric else [max(0.0, center - value), center + value]

    def augment(self, x):
        rgb = x[..., -3:].reshape(-1, 3)

        self.apply_brightness(rgb)
        self.apply_contrast(rgb)
        self.apply_saturation(rgb)

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
            x[t, drop_mask] = x[t, 0]


class PointCoordinateNoise(Augmentation):
    """Clipped Gaussian noise on all XYZ coordinates, in raw coordinate units."""

    def __init__(self, noise_std=0.002, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std
        if not np.isfinite((self.noise_std, self.clip_range)).all() or min(self.noise_std, self.clip_range) < 0:
            raise ValueError("noise_std/clip_range must be finite and non-negative")

    def augment(self, x):
        if self.noise_std <= 0:
            return

        xyz = x[..., :3]
        xyz += clipped_gaussian_noise(xyz.shape, self.noise_std, self.clip_range, x.dtype)


class StateNoise(Augmentation):
    """Noise is applied in raw state units."""

    def __init__(self, noise_std=0.0002, clip_range=None, prob=1.0):
        super().__init__(prob=prob)
        self.noise_std = float(noise_std)
        self.clip_range = float(clip_range) if clip_range is not None else 2.0 * self.noise_std
        if not np.isfinite((self.noise_std, self.clip_range)).all() or min(self.noise_std, self.clip_range) < 0:
            raise ValueError("noise_std/clip_range must be finite and non-negative")

    def augment(self, x):
        x += clipped_gaussian_noise(x.shape, self.noise_std, self.clip_range, x.dtype)


class ImageAugmentation:
    """Color jitter and grayscale on float32 observation windows."""

    def __init__(
        self,
        brightness=0.6,
        contrast=0.3,
        saturation=0.3,
        hue=0.08,
        grayscale_prob=0.2,
        prob=1.0,
    ):
        import torchvision.transforms.v2 as v2

        if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
               for value in (brightness, contrast, saturation, hue)):
            raise ValueError("color strengths must be finite non-negative scalars")
        if not np.isfinite((grayscale_prob, prob)).all() or not 0 <= grayscale_prob <= 1 or not 0 <= prob <= 1:
            raise ValueError("RGB augmentation probabilities must be finite in [0,1]")
        operations = []
        if prob > 0:
            if any((brightness, contrast, saturation, hue)):
                operations.append(v2.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                ))
            if grayscale_prob > 0:
                operations.append(v2.RandomGrayscale(p=grayscale_prob))
        self.transform = v2.Compose(operations) if operations else torch.nn.Identity()
        if operations and 0 < prob < 1:
            self.transform = v2.RandomApply([self.transform], p=prob)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Transform decisions and color parameters are shared across the T-frame window.
        return self.transform(x).clamp(0, 1)


# NumPy transforms run in this order before RGB processing.
AUGMENTATION_SPECS = (
    ("pc", PointCoordinateNoise, "coord_noise", "point_cloud"),
    ("pc", PointColorJitter, "color", "point_cloud"),
    ("pc", PointDropout, "dropout", "point_cloud"),
    ("state", StateNoise, "noise", "joint_state"),
)
