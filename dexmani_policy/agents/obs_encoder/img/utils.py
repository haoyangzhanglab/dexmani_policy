import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_channel_first(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] in (1, 3) and x.shape[-3] not in (1, 3):
        return x.moveaxis(-1, -3)
    return x


def to_float01(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.max() > 1.0:
        x = x / 255.0
    return x


def resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


def center_crop(x: torch.Tensor, size) -> torch.Tensor:
    h, w = size
    H, W = x.shape[-2:]
    top = (H - h) // 2
    left = (W - w) // 2
    return x[..., top:top + h, left:left + w]


def random_crop(x: torch.Tensor, size) -> torch.Tensor:
    h, w = size
    n, _, H, W = x.shape
    max_top = H - h
    max_left = W - w
    tops = torch.randint(0, max_top + 1, (n,), device=x.device)
    lefts = torch.randint(0, max_left + 1, (n,), device=x.device)
    return torch.cat([
        x[i:i + 1, :, tops[i]:tops[i] + h, lefts[i]:lefts[i] + w]
        for i in range(n)
    ], dim=0)


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean = x.new_tensor(mean).view(1, -1, 1, 1)
    std = x.new_tensor(std).view(1, -1, 1, 1)
    return (x - mean) / std


def random_color_jitter(
    x: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
) -> torch.Tensor:
    if brightness > 0:
        scale = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * brightness
        x = x * scale
    if contrast > 0:
        scale = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * contrast
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * scale + mean
    if saturation > 0:
        scale = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * saturation
        gray = x.mean(dim=1, keepdim=True)
        x = (x - gray) * scale + gray
    return x.clamp(0.0, 1.0)


def random_blur(x: torch.Tensor, p: float = 0.0, kernel_size: int = 3) -> torch.Tensor:
    if p <= 0:
        return x
    mask = torch.rand(x.shape[0], device=x.device) < p
    if not mask.any():
        return x
    pad = kernel_size // 2
    y = F.avg_pool2d(F.pad(x[mask], (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
    x = x.clone()
    x[mask] = y
    return x


def random_noise(x: torch.Tensor, std: float = 0.0) -> torch.Tensor:
    if std <= 0:
        return x
    return (x + torch.randn_like(x) * std).clamp(0.0, 1.0)


class ImageTransform(nn.Module):
    """
    Minimal image preprocessor for imitation learning.

    Input:
        HWC / BHWC / BTHWC / CHW / BCHW / BTCHW
    Output:
        same leading dimensions, always channel-first
    """

    def __init__(
        self,
        resize_shape=None,
        crop_shape=None,
        random_crop: bool = True,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        blur_prob: float = 0.0,
        blur_kernel_size: int = 3,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.random_crop = random_crop
        self.mean = mean
        self.std = std
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blur_prob = blur_prob
        self.blur_kernel_size = blur_kernel_size
        self.noise_std = noise_std

    def _flatten(self, x: torch.Tensor):
        x = to_channel_first(x)
        leading_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        return x, leading_shape

    def _restore(self, x: torch.Tensor, leading_shape):
        if len(leading_shape) == 0:
            return x[0]
        return x.reshape(*leading_shape, *x.shape[-3:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, leading_shape = self._flatten(x)
        x = to_float01(x)

        if self.resize_shape is not None:
            x = resize(x, self.resize_shape)

        if self.crop_shape is not None:
            if self.training and self.random_crop:
                x = random_crop(x, self.crop_shape)
            else:
                x = center_crop(x, self.crop_shape)

        if self.training:
            x = random_color_jitter(
                x,
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
            )
            x = random_blur(x, p=self.blur_prob, kernel_size=self.blur_kernel_size)
            x = random_noise(x, std=self.noise_std)

        if self.mean is not None and self.std is not None:
            x = normalize(x, self.mean, self.std)

        return self._restore(x, leading_shape)



def make_default_image_transform(resize_shape=None, crop_shape=None):
    return ImageTransform(
        resize_shape=resize_shape,
        crop_shape=crop_shape,
        random_crop=True,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        blur_prob=0.1,
        blur_kernel_size=3,
        noise_std=0.01,
    )


def example():
    torch.manual_seed(0)

    x = torch.randint(0, 256, (2, 3, 480, 640, 3), dtype=torch.uint8)
    transform = ImageTransform(
        resize_shape=(240, 240),
        crop_shape=(224, 224),
        random_crop=True,
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        blur_prob=0.1,
        noise_std=0.01,
    )

    transform.train()
    y_train = transform(x)
    print("train input shape: ", tuple(x.shape))
    print("train output shape:", tuple(y_train.shape))
    print("train output dtype:", y_train.dtype)

    transform.eval()
    y_eval = transform(x)
    print("eval output shape: ", tuple(y_eval.shape))
    print("eval output dtype: ", y_eval.dtype)


if __name__ == "__main__":
    example()