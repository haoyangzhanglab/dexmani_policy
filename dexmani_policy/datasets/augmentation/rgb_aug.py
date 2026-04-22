import numpy as np
from PIL import Image
import torchvision.transforms as T


class RGBAug:
    def __init__(self, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.05):
        self.jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(self, images):
        # (T, H, W, 3) uint8 numpy → same shape/dtype; per-frame independent aug by design
        return np.stack([np.array(self.jitter(Image.fromarray(f))) for f in images])
