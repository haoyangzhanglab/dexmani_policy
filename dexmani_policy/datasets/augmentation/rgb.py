import torch
import torchvision.transforms.v2 as v2


class RGBAug:
    """Color + noise + blur augmentation for pre-resized float32 CHW tensors.

    Designed to run AFTER resize in ``__getitem__`` so the augmentation
    operates on small 224×224 tensors with no dtype / layout roundtrip.

    Each sub-augmentation is independently probabilistic: set its prob
    < 1.0 to apply it only on a fraction of samples.  Set its strength
    param to 0 to disable it entirely.
    """

    def __init__(
        self,
        # ColorJitter — 参考 DexUMI 的激进参数 (brightness=0.64 应对光照变化)
        brightness=0.6,
        contrast=0.3,
        saturation=0.3,
        hue=0.08,
        # Grayscale (逼模型依赖形状而非颜色)
        grayscale_prob=0.2,
        # GaussianNoise  (0 = disabled)
        noise_std=0.0,
        noise_prob=1.0,
        # GaussianBlur (模拟运动模糊/失焦, DexUMI: kernel=3, p=0.2)
        blur_kernel=3,
        blur_sigma=(0.1, 2.0),
        blur_prob=0.2,
        # Global: 每帧都做增强  (DexUMI: p=1.0)
        prob=1.0,
    ):
        self.prob = prob

        self.color_jitter = v2.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )
        self.grayscale = v2.RandomGrayscale(p=grayscale_prob) if grayscale_prob > 0 else None

        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.noise = v2.GaussianNoise(mean=0.0, sigma=noise_std) if noise_std > 0 else None

        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.blur_prob = blur_prob
        self.blur = v2.GaussianBlur(
            kernel_size=blur_kernel, sigma=blur_sigma
        ) if blur_kernel > 0 else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, 3, H, W) float32 [0, 1] → same shape."""
        if self.prob < 1.0 and torch.rand(1).item() > self.prob:
            return x

        x = self.color_jitter(x)

        if self.grayscale is not None:
            x = self.grayscale(x)

        if self.noise is not None and torch.rand(1).item() <= self.noise_prob:
            x = self.noise(x)

        if self.blur is not None and torch.rand(1).item() <= self.blur_prob:
            x = self.blur(x)

        return x.clamp_(0, 1)
