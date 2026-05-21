import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter
from .base import Aug


class RGBAug(Aug):
    """ColorJitter for RGB images with temporal consistency.

    Reference: Diffusion Policy (RSS 2023) — same jitter params across all T frames.
    """

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05,
                 prob=1.0):
        super().__init__(prob=prob)
        # Use ColorJitter to normalise param ranges, then read back the list form
        cj = ColorJitter(brightness=brightness, contrast=contrast,
                         saturation=saturation, hue=hue)
        self.b_range = cj.brightness
        self.c_range = cj.contrast
        self.s_range = cj.saturation
        self.h_range = cj.hue

    def _augment(self, x):
        # Sample one set of params, apply to all T frames (temporal consistency)
        params = ColorJitter.get_params(self.b_range, self.c_range, self.s_range, self.h_range)
        return np.stack([self.apply_transform(params, Image.fromarray(f)) for f in x])

    @staticmethod
    def apply_transform(params, img):
        fn_idx, b, c, s, h = params
        for fn_id in fn_idx.tolist():
            if fn_id == 0:
                img = F.adjust_brightness(img, b)
            elif fn_id == 1:
                img = F.adjust_contrast(img, c)
            elif fn_id == 2:
                img = F.adjust_saturation(img, s)
            elif fn_id == 3:
                img = F.adjust_hue(img, h)
        return np.array(img)
